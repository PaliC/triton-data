class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "i64[1, 512]", arg2_1: "f32[50265, 768]", arg3_1: "f32[2, 768]", arg4_1: "f32[512, 768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768, 768]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768, 768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[3072, 768]", arg18_1: "f32[3072]", arg19_1: "f32[768, 3072]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768, 768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768, 768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[3072, 768]", arg34_1: "f32[3072]", arg35_1: "f32[768, 3072]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768, 768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[3072, 768]", arg50_1: "f32[3072]", arg51_1: "f32[768, 3072]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768, 768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768, 768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[3072, 768]", arg66_1: "f32[3072]", arg67_1: "f32[768, 3072]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768, 768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768, 768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[768]", arg81_1: "f32[3072, 768]", arg82_1: "f32[3072]", arg83_1: "f32[768, 3072]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[768]", arg87_1: "f32[768, 768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[3072, 768]", arg98_1: "f32[3072]", arg99_1: "f32[768, 3072]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[768, 768]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768, 768]", arg108_1: "f32[768]", arg109_1: "f32[768, 768]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[3072, 768]", arg114_1: "f32[3072]", arg115_1: "f32[768, 3072]", arg116_1: "f32[768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[768, 768]", arg120_1: "f32[768]", arg121_1: "f32[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[768, 768]", arg124_1: "f32[768]", arg125_1: "f32[768, 768]", arg126_1: "f32[768]", arg127_1: "f32[768]", arg128_1: "f32[768]", arg129_1: "f32[3072, 768]", arg130_1: "f32[3072]", arg131_1: "f32[768, 3072]", arg132_1: "f32[768]", arg133_1: "f32[768]", arg134_1: "f32[768]", arg135_1: "f32[768, 768]", arg136_1: "f32[768]", arg137_1: "f32[768, 768]", arg138_1: "f32[768]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[3072, 768]", arg146_1: "f32[3072]", arg147_1: "f32[768, 3072]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[768, 768]", arg152_1: "f32[768]", arg153_1: "f32[768, 768]", arg154_1: "f32[768]", arg155_1: "f32[768, 768]", arg156_1: "f32[768]", arg157_1: "f32[768, 768]", arg158_1: "f32[768]", arg159_1: "f32[768]", arg160_1: "f32[768]", arg161_1: "f32[3072, 768]", arg162_1: "f32[3072]", arg163_1: "f32[768, 3072]", arg164_1: "f32[768]", arg165_1: "f32[768]", arg166_1: "f32[768]", arg167_1: "f32[768, 768]", arg168_1: "f32[768]", arg169_1: "f32[768, 768]", arg170_1: "f32[768]", arg171_1: "f32[768, 768]", arg172_1: "f32[768]", arg173_1: "f32[768, 768]", arg174_1: "f32[768]", arg175_1: "f32[768]", arg176_1: "f32[768]", arg177_1: "f32[3072, 768]", arg178_1: "f32[3072]", arg179_1: "f32[768, 3072]", arg180_1: "f32[768]", arg181_1: "f32[768]", arg182_1: "f32[768]", arg183_1: "f32[768, 768]", arg184_1: "f32[768]", arg185_1: "f32[768, 768]", arg186_1: "f32[768]", arg187_1: "f32[768, 768]", arg188_1: "f32[768]", arg189_1: "f32[768, 768]", arg190_1: "f32[768]", arg191_1: "f32[768]", arg192_1: "f32[768]", arg193_1: "f32[3072, 768]", arg194_1: "f32[3072]", arg195_1: "f32[768, 3072]", arg196_1: "f32[768]", arg197_1: "f32[768]", arg198_1: "f32[768]", arg199_1: "f32[2, 768]", arg200_1: "f32[2]", arg201_1: "i64[16]", arg202_1: "i64[16]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:907 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[16, 512]" = torch.ops.aten.expand.default(arg1_1, [16, 512]);  arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1701 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne: "b8[16, 512]" = torch.ops.aten.ne.Scalar(arg0_1, 0)
        convert_element_type: "i32[16, 512]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1702 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        cumsum: "i64[16, 512]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
        convert_element_type_1: "i32[16, 512]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add: "i32[16, 512]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
        mul: "i32[16, 512]" = torch.ops.aten.mul.Tensor(add, convert_element_type);  add = convert_element_type = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1703 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_2: "i64[16, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add_1: "i64[16, 512]" = torch.ops.aten.add.Tensor(convert_element_type_2, 0);  convert_element_type_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:122 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[16, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, arg0_1, 0);  arg2_1 = arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:123 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[16, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, expand);  arg3_1 = expand = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:125 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add_2: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:127 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[16, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, add_1, 0);  arg4_1 = add_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:128 in forward, code: embeddings += position_embeddings
        add_3: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_2);  add_2 = embedding_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:129 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean[0]
        getitem_1: "f32[16, 512, 1]" = var_mean[1];  var_mean = None
        add_4: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_1);  add_3 = getitem_1 = None
        mul_1: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        add_5: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg6_1);  mul_2 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:921 in forward, code: attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        full: "f32[16, 512]" = torch.ops.aten.full.default([16, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:184 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze: "f32[16, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1: "f32[16, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1: "f32[16, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_1, [16, 1, 512, 512]);  unsqueeze_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:186 in _expand_mask, code: inverted_mask = 1.0 - expanded_mask
        sub_1: "f32[16, 1, 512, 512]" = torch.ops.aten.sub.Tensor(1.0, expand_1);  expand_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:188 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type_3: "b8[16, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where: "f32[16, 1, 512, 512]" = torch.ops.aten.where.self(convert_element_type_3, scalar_tensor, sub_1);  convert_element_type_3 = scalar_tensor = sub_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view: "f32[8192, 768]" = torch.ops.aten.view.default(add_5, [8192, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg8_1, view, permute);  arg8_1 = view = permute = None
        view_1: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm, [16, 512, 768]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_2: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_1, [16, 512, 12, 64]);  view_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_1: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_3: "f32[8192, 768]" = torch.ops.aten.view.default(add_5, [8192, 768])
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg10_1, view_3, permute_2);  arg10_1 = view_3 = permute_2 = None
        view_4: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_1, [16, 512, 768]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_5: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_4, [16, 512, 12, 64]);  view_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_3: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_6: "f32[8192, 768]" = torch.ops.aten.view.default(add_5, [8192, 768])
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg12_1, view_6, permute_4);  arg12_1 = view_6 = permute_4 = None
        view_7: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_2, [16, 512, 768]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_8: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_7, [16, 512, 12, 64]);  view_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_5: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_2, False);  permute_1 = permute_3 = permute_5 = expand_2 = None
        getitem_2: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_6: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_9: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_6, [16, 512, 768]);  permute_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_10: "f32[8192, 768]" = torch.ops.aten.view.default(view_9, [8192, 768]);  view_9 = None
        permute_7: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_3: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg14_1, view_10, permute_7);  arg14_1 = view_10 = permute_7 = None
        view_11: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_3, [16, 512, 768]);  addmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_6: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_11, add_5);  view_11 = add_5 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_6: "f32[16, 512, 1]" = var_mean_1[0]
        getitem_7: "f32[16, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_2: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_7);  add_6 = getitem_7 = None
        mul_3: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_4: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg15_1);  mul_3 = arg15_1 = None
        add_8: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_12: "f32[8192, 768]" = torch.ops.aten.view.default(add_8, [8192, 768])
        permute_8: "f32[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_4: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_12, permute_8);  arg18_1 = view_12 = permute_8 = None
        view_13: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_4, [16, 512, 3072]);  addmm_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_5: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_6: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_9: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_9);  mul_5 = add_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_14: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_7, [8192, 3072]);  mul_7 = None
        permute_9: "f32[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_5: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg20_1, view_14, permute_9);  arg20_1 = view_14 = permute_9 = None
        view_15: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_5, [16, 512, 768]);  addmm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_10: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_15, add_8);  view_15 = add_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_8: "f32[16, 512, 1]" = var_mean_2[0]
        getitem_9: "f32[16, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_3: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_9);  add_10 = getitem_9 = None
        mul_8: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_9: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg21_1);  mul_8 = arg21_1 = None
        add_12: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_16: "f32[8192, 768]" = torch.ops.aten.view.default(add_12, [8192, 768])
        permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg24_1, view_16, permute_10);  arg24_1 = view_16 = permute_10 = None
        view_17: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_6, [16, 512, 768]);  addmm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_18: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_17, [16, 512, 12, 64]);  view_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_11: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_19: "f32[8192, 768]" = torch.ops.aten.view.default(add_12, [8192, 768])
        permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_7: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg26_1, view_19, permute_12);  arg26_1 = view_19 = permute_12 = None
        view_20: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_7, [16, 512, 768]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_21: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_20, [16, 512, 12, 64]);  view_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_13: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_22: "f32[8192, 768]" = torch.ops.aten.view.default(add_12, [8192, 768])
        permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_8: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg28_1, view_22, permute_14);  arg28_1 = view_22 = permute_14 = None
        view_23: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_8, [16, 512, 768]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_24: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_23, [16, 512, 12, 64]);  view_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_15: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_3: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_3, False);  permute_11 = permute_13 = permute_15 = expand_3 = None
        getitem_10: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_16: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_25: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_16, [16, 512, 768]);  permute_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_26: "f32[8192, 768]" = torch.ops.aten.view.default(view_25, [8192, 768]);  view_25 = None
        permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_9: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg30_1, view_26, permute_17);  arg30_1 = view_26 = permute_17 = None
        view_27: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_9, [16, 512, 768]);  addmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_13: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_27, add_12);  view_27 = add_12 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_14: "f32[16, 512, 1]" = var_mean_3[0]
        getitem_15: "f32[16, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_14: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_4: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_15);  add_13 = getitem_15 = None
        mul_10: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_11: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg31_1);  mul_10 = arg31_1 = None
        add_15: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg32_1);  mul_11 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_28: "f32[8192, 768]" = torch.ops.aten.view.default(add_15, [8192, 768])
        permute_18: "f32[768, 3072]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_10: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg34_1, view_28, permute_18);  arg34_1 = view_28 = permute_18 = None
        view_29: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [16, 512, 3072]);  addmm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_12: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_13: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_16: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_30: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_14, [8192, 3072]);  mul_14 = None
        permute_19: "f32[3072, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_11: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg36_1, view_30, permute_19);  arg36_1 = view_30 = permute_19 = None
        view_31: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_11, [16, 512, 768]);  addmm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_17: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_31, add_15);  view_31 = add_15 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_16: "f32[16, 512, 1]" = var_mean_4[0]
        getitem_17: "f32[16, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_5: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_17);  add_17 = getitem_17 = None
        mul_15: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_16: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg37_1);  mul_15 = arg37_1 = None
        add_19: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, arg38_1);  mul_16 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_32: "f32[8192, 768]" = torch.ops.aten.view.default(add_19, [8192, 768])
        permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_12: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg40_1, view_32, permute_20);  arg40_1 = view_32 = permute_20 = None
        view_33: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_12, [16, 512, 768]);  addmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_34: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_33, [16, 512, 12, 64]);  view_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_21: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_35: "f32[8192, 768]" = torch.ops.aten.view.default(add_19, [8192, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_13: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg42_1, view_35, permute_22);  arg42_1 = view_35 = permute_22 = None
        view_36: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_13, [16, 512, 768]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_37: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_36, [16, 512, 12, 64]);  view_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_23: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_38: "f32[8192, 768]" = torch.ops.aten.view.default(add_19, [8192, 768])
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_14: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg44_1, view_38, permute_24);  arg44_1 = view_38 = permute_24 = None
        view_39: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_14, [16, 512, 768]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_40: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_39, [16, 512, 12, 64]);  view_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_25: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_4, False);  permute_21 = permute_23 = permute_25 = expand_4 = None
        getitem_18: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_26: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_41: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_26, [16, 512, 768]);  permute_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[8192, 768]" = torch.ops.aten.view.default(view_41, [8192, 768]);  view_41 = None
        permute_27: "f32[768, 768]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_15: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg46_1, view_42, permute_27);  arg46_1 = view_42 = permute_27 = None
        view_43: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_15, [16, 512, 768]);  addmm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_20: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_43, add_19);  view_43 = add_19 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_22: "f32[16, 512, 1]" = var_mean_5[0]
        getitem_23: "f32[16, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_6: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_23);  add_20 = getitem_23 = None
        mul_17: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_18: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg47_1);  mul_17 = arg47_1 = None
        add_22: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_18, arg48_1);  mul_18 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_44: "f32[8192, 768]" = torch.ops.aten.view.default(add_22, [8192, 768])
        permute_28: "f32[768, 3072]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_16: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg50_1, view_44, permute_28);  arg50_1 = view_44 = permute_28 = None
        view_45: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_16, [16, 512, 3072]);  addmm_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_19: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_20: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_23: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_23);  mul_19 = add_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_46: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_21, [8192, 3072]);  mul_21 = None
        permute_29: "f32[3072, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_17: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg52_1, view_46, permute_29);  arg52_1 = view_46 = permute_29 = None
        view_47: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_17, [16, 512, 768]);  addmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_24: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_47, add_22);  view_47 = add_22 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_24: "f32[16, 512, 1]" = var_mean_6[0]
        getitem_25: "f32[16, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_7: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_25);  add_24 = getitem_25 = None
        mul_22: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_23: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg53_1);  mul_22 = arg53_1 = None
        add_26: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, arg54_1);  mul_23 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_48: "f32[8192, 768]" = torch.ops.aten.view.default(add_26, [8192, 768])
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg56_1, view_48, permute_30);  arg56_1 = view_48 = permute_30 = None
        view_49: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_18, [16, 512, 768]);  addmm_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_50: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_49, [16, 512, 12, 64]);  view_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_31: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_51: "f32[8192, 768]" = torch.ops.aten.view.default(add_26, [8192, 768])
        permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_19: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg58_1, view_51, permute_32);  arg58_1 = view_51 = permute_32 = None
        view_52: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_19, [16, 512, 768]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_53: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_52, [16, 512, 12, 64]);  view_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_33: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_54: "f32[8192, 768]" = torch.ops.aten.view.default(add_26, [8192, 768])
        permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_20: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg60_1, view_54, permute_34);  arg60_1 = view_54 = permute_34 = None
        view_55: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_20, [16, 512, 768]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_56: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_55, [16, 512, 12, 64]);  view_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_35: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_5: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_5, False);  permute_31 = permute_33 = permute_35 = expand_5 = None
        getitem_26: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_36: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_57: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_36, [16, 512, 768]);  permute_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_58: "f32[8192, 768]" = torch.ops.aten.view.default(view_57, [8192, 768]);  view_57 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_21: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg62_1, view_58, permute_37);  arg62_1 = view_58 = permute_37 = None
        view_59: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_21, [16, 512, 768]);  addmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_59, add_26);  view_59 = add_26 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_30: "f32[16, 512, 1]" = var_mean_7[0]
        getitem_31: "f32[16, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_28: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_8: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_31);  add_27 = getitem_31 = None
        mul_24: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_25: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg63_1);  mul_24 = arg63_1 = None
        add_29: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, arg64_1);  mul_25 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[8192, 768]" = torch.ops.aten.view.default(add_29, [8192, 768])
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_22: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg66_1, view_60, permute_38);  arg66_1 = view_60 = permute_38 = None
        view_61: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [16, 512, 3072]);  addmm_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_26: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_27: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_30: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_30);  mul_26 = add_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_28, [8192, 3072]);  mul_28 = None
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_23: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg68_1, view_62, permute_39);  arg68_1 = view_62 = permute_39 = None
        view_63: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_23, [16, 512, 768]);  addmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_31: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_63, add_29);  view_63 = add_29 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_32: "f32[16, 512, 1]" = var_mean_8[0]
        getitem_33: "f32[16, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_32: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_9: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_33);  add_31 = getitem_33 = None
        mul_29: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_30: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, arg69_1);  mul_29 = arg69_1 = None
        add_33: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_30, arg70_1);  mul_30 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_64: "f32[8192, 768]" = torch.ops.aten.view.default(add_33, [8192, 768])
        permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_24: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg72_1, view_64, permute_40);  arg72_1 = view_64 = permute_40 = None
        view_65: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_24, [16, 512, 768]);  addmm_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_66: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_65, [16, 512, 12, 64]);  view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_41: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_67: "f32[8192, 768]" = torch.ops.aten.view.default(add_33, [8192, 768])
        permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_25: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg74_1, view_67, permute_42);  arg74_1 = view_67 = permute_42 = None
        view_68: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_25, [16, 512, 768]);  addmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_69: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_68, [16, 512, 12, 64]);  view_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_43: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_70: "f32[8192, 768]" = torch.ops.aten.view.default(add_33, [8192, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_26: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg76_1, view_70, permute_44);  arg76_1 = view_70 = permute_44 = None
        view_71: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_26, [16, 512, 768]);  addmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_72: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_71, [16, 512, 12, 64]);  view_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_45: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_6, False);  permute_41 = permute_43 = permute_45 = expand_6 = None
        getitem_34: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_46: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_73: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_46, [16, 512, 768]);  permute_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_74: "f32[8192, 768]" = torch.ops.aten.view.default(view_73, [8192, 768]);  view_73 = None
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_27: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg78_1, view_74, permute_47);  arg78_1 = view_74 = permute_47 = None
        view_75: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_27, [16, 512, 768]);  addmm_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_34: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_75, add_33);  view_75 = add_33 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
        getitem_38: "f32[16, 512, 1]" = var_mean_9[0]
        getitem_39: "f32[16, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_35: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_10: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_34, getitem_39);  add_34 = getitem_39 = None
        mul_31: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_32: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, arg79_1);  mul_31 = arg79_1 = None
        add_36: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, arg80_1);  mul_32 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_76: "f32[8192, 768]" = torch.ops.aten.view.default(add_36, [8192, 768])
        permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_28: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg82_1, view_76, permute_48);  arg82_1 = view_76 = permute_48 = None
        view_77: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_28, [16, 512, 3072]);  addmm_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_33: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_34: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_37: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_37);  mul_33 = add_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_78: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_35, [8192, 3072]);  mul_35 = None
        permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_29: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg84_1, view_78, permute_49);  arg84_1 = view_78 = permute_49 = None
        view_79: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_29, [16, 512, 768]);  addmm_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_38: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_79, add_36);  view_79 = add_36 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_40: "f32[16, 512, 1]" = var_mean_10[0]
        getitem_41: "f32[16, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_39: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_11: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_41);  add_38 = getitem_41 = None
        mul_36: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_37: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg85_1);  mul_36 = arg85_1 = None
        add_40: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg86_1);  mul_37 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_80: "f32[8192, 768]" = torch.ops.aten.view.default(add_40, [8192, 768])
        permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_30: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg88_1, view_80, permute_50);  arg88_1 = view_80 = permute_50 = None
        view_81: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_30, [16, 512, 768]);  addmm_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_82: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_81, [16, 512, 12, 64]);  view_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_51: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_83: "f32[8192, 768]" = torch.ops.aten.view.default(add_40, [8192, 768])
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_31: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg90_1, view_83, permute_52);  arg90_1 = view_83 = permute_52 = None
        view_84: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_31, [16, 512, 768]);  addmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_85: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_84, [16, 512, 12, 64]);  view_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_53: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_86: "f32[8192, 768]" = torch.ops.aten.view.default(add_40, [8192, 768])
        permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_32: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg92_1, view_86, permute_54);  arg92_1 = view_86 = permute_54 = None
        view_87: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_32, [16, 512, 768]);  addmm_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_88: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_87, [16, 512, 12, 64]);  view_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_55: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_7: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_7, False);  permute_51 = permute_53 = permute_55 = expand_7 = None
        getitem_42: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_56: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_89: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_56, [16, 512, 768]);  permute_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_90: "f32[8192, 768]" = torch.ops.aten.view.default(view_89, [8192, 768]);  view_89 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_33: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg94_1, view_90, permute_57);  arg94_1 = view_90 = permute_57 = None
        view_91: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_33, [16, 512, 768]);  addmm_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_41: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_91, add_40);  view_91 = add_40 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_46: "f32[16, 512, 1]" = var_mean_11[0]
        getitem_47: "f32[16, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_42: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_12: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_47);  add_41 = getitem_47 = None
        mul_38: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_39: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg95_1);  mul_38 = arg95_1 = None
        add_43: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_39, arg96_1);  mul_39 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_92: "f32[8192, 768]" = torch.ops.aten.view.default(add_43, [8192, 768])
        permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_34: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg98_1, view_92, permute_58);  arg98_1 = view_92 = permute_58 = None
        view_93: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_34, [16, 512, 3072]);  addmm_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_40: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_41: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_44: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_44);  mul_40 = add_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_94: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_42, [8192, 3072]);  mul_42 = None
        permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_35: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg100_1, view_94, permute_59);  arg100_1 = view_94 = permute_59 = None
        view_95: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_35, [16, 512, 768]);  addmm_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_45: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_95, add_43);  view_95 = add_43 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_48: "f32[16, 512, 1]" = var_mean_12[0]
        getitem_49: "f32[16, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_46: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_13: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_49);  add_45 = getitem_49 = None
        mul_43: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_44: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, arg101_1);  mul_43 = arg101_1 = None
        add_47: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_44, arg102_1);  mul_44 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_96: "f32[8192, 768]" = torch.ops.aten.view.default(add_47, [8192, 768])
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_36: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg104_1, view_96, permute_60);  arg104_1 = view_96 = permute_60 = None
        view_97: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_36, [16, 512, 768]);  addmm_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_98: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_97, [16, 512, 12, 64]);  view_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_61: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_99: "f32[8192, 768]" = torch.ops.aten.view.default(add_47, [8192, 768])
        permute_62: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_37: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg106_1, view_99, permute_62);  arg106_1 = view_99 = permute_62 = None
        view_100: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_37, [16, 512, 768]);  addmm_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_101: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_100, [16, 512, 12, 64]);  view_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_63: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_102: "f32[8192, 768]" = torch.ops.aten.view.default(add_47, [8192, 768])
        permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_38: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg108_1, view_102, permute_64);  arg108_1 = view_102 = permute_64 = None
        view_103: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_38, [16, 512, 768]);  addmm_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_104: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_103, [16, 512, 12, 64]);  view_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_65: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_63, permute_65, expand_8, False);  permute_61 = permute_63 = permute_65 = expand_8 = None
        getitem_50: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_66: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_105: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_66, [16, 512, 768]);  permute_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[8192, 768]" = torch.ops.aten.view.default(view_105, [8192, 768]);  view_105 = None
        permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_39: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg110_1, view_106, permute_67);  arg110_1 = view_106 = permute_67 = None
        view_107: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_39, [16, 512, 768]);  addmm_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_48: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_107, add_47);  view_107 = add_47 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_54: "f32[16, 512, 1]" = var_mean_13[0]
        getitem_55: "f32[16, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_49: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_13: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_14: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_55);  add_48 = getitem_55 = None
        mul_45: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_46: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg111_1);  mul_45 = arg111_1 = None
        add_50: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, arg112_1);  mul_46 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[8192, 768]" = torch.ops.aten.view.default(add_50, [8192, 768])
        permute_68: "f32[768, 3072]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_40: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg114_1, view_108, permute_68);  arg114_1 = view_108 = permute_68 = None
        view_109: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_40, [16, 512, 3072]);  addmm_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_47: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_48: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_51: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_51);  mul_47 = add_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_110: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_49, [8192, 3072]);  mul_49 = None
        permute_69: "f32[3072, 768]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_41: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg116_1, view_110, permute_69);  arg116_1 = view_110 = permute_69 = None
        view_111: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_41, [16, 512, 768]);  addmm_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_52: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_111, add_50);  view_111 = add_50 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_56: "f32[16, 512, 1]" = var_mean_14[0]
        getitem_57: "f32[16, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_53: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_14: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_15: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_57);  add_52 = getitem_57 = None
        mul_50: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_51: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg117_1);  mul_50 = arg117_1 = None
        add_54: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_51, arg118_1);  mul_51 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_112: "f32[8192, 768]" = torch.ops.aten.view.default(add_54, [8192, 768])
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_42: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg120_1, view_112, permute_70);  arg120_1 = view_112 = permute_70 = None
        view_113: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_42, [16, 512, 768]);  addmm_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_114: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_113, [16, 512, 12, 64]);  view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_71: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_115: "f32[8192, 768]" = torch.ops.aten.view.default(add_54, [8192, 768])
        permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_43: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg122_1, view_115, permute_72);  arg122_1 = view_115 = permute_72 = None
        view_116: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_43, [16, 512, 768]);  addmm_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_117: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_116, [16, 512, 12, 64]);  view_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_73: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_118: "f32[8192, 768]" = torch.ops.aten.view.default(add_54, [8192, 768])
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_44: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg124_1, view_118, permute_74);  arg124_1 = view_118 = permute_74 = None
        view_119: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_44, [16, 512, 768]);  addmm_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_120: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_119, [16, 512, 12, 64]);  view_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_75: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_9: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_73, permute_75, expand_9, False);  permute_71 = permute_73 = permute_75 = expand_9 = None
        getitem_58: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_76: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_121: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_76, [16, 512, 768]);  permute_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_122: "f32[8192, 768]" = torch.ops.aten.view.default(view_121, [8192, 768]);  view_121 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_45: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg126_1, view_122, permute_77);  arg126_1 = view_122 = permute_77 = None
        view_123: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_45, [16, 512, 768]);  addmm_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_55: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_123, add_54);  view_123 = add_54 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_62: "f32[16, 512, 1]" = var_mean_15[0]
        getitem_63: "f32[16, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_56: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_15: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_16: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_63);  add_55 = getitem_63 = None
        mul_52: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_53: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, arg127_1);  mul_52 = arg127_1 = None
        add_57: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_53, arg128_1);  mul_53 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_124: "f32[8192, 768]" = torch.ops.aten.view.default(add_57, [8192, 768])
        permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_46: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg130_1, view_124, permute_78);  arg130_1 = view_124 = permute_78 = None
        view_125: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_46, [16, 512, 3072]);  addmm_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_54: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_55: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_58: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_54, add_58);  mul_54 = add_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_56, [8192, 3072]);  mul_56 = None
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_47: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg132_1, view_126, permute_79);  arg132_1 = view_126 = permute_79 = None
        view_127: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_47, [16, 512, 768]);  addmm_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_59: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_127, add_57);  view_127 = add_57 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_64: "f32[16, 512, 1]" = var_mean_16[0]
        getitem_65: "f32[16, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_60: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_16: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_17: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_65);  add_59 = getitem_65 = None
        mul_57: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_58: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg133_1);  mul_57 = arg133_1 = None
        add_61: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_58, arg134_1);  mul_58 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_128: "f32[8192, 768]" = torch.ops.aten.view.default(add_61, [8192, 768])
        permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_48: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg136_1, view_128, permute_80);  arg136_1 = view_128 = permute_80 = None
        view_129: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_48, [16, 512, 768]);  addmm_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_130: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_129, [16, 512, 12, 64]);  view_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_81: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_131: "f32[8192, 768]" = torch.ops.aten.view.default(add_61, [8192, 768])
        permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_49: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg138_1, view_131, permute_82);  arg138_1 = view_131 = permute_82 = None
        view_132: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_49, [16, 512, 768]);  addmm_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_133: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_132, [16, 512, 12, 64]);  view_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_83: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_134: "f32[8192, 768]" = torch.ops.aten.view.default(add_61, [8192, 768])
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_50: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg140_1, view_134, permute_84);  arg140_1 = view_134 = permute_84 = None
        view_135: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_50, [16, 512, 768]);  addmm_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_136: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_135, [16, 512, 12, 64]);  view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_85: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_81, permute_83, permute_85, expand_10, False);  permute_81 = permute_83 = permute_85 = expand_10 = None
        getitem_66: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_86: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_137: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_86, [16, 512, 768]);  permute_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_138: "f32[8192, 768]" = torch.ops.aten.view.default(view_137, [8192, 768]);  view_137 = None
        permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_51: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg142_1, view_138, permute_87);  arg142_1 = view_138 = permute_87 = None
        view_139: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_51, [16, 512, 768]);  addmm_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_62: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_139, add_61);  view_139 = add_61 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_70: "f32[16, 512, 1]" = var_mean_17[0]
        getitem_71: "f32[16, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_63: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_17: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_18: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_62, getitem_71);  add_62 = getitem_71 = None
        mul_59: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_60: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, arg143_1);  mul_59 = arg143_1 = None
        add_64: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_60, arg144_1);  mul_60 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_140: "f32[8192, 768]" = torch.ops.aten.view.default(add_64, [8192, 768])
        permute_88: "f32[768, 3072]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_52: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg146_1, view_140, permute_88);  arg146_1 = view_140 = permute_88 = None
        view_141: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_52, [16, 512, 3072]);  addmm_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_61: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_62: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_65: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_61, add_65);  mul_61 = add_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_142: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_63, [8192, 3072]);  mul_63 = None
        permute_89: "f32[3072, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_53: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg148_1, view_142, permute_89);  arg148_1 = view_142 = permute_89 = None
        view_143: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_53, [16, 512, 768]);  addmm_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_66: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_143, add_64);  view_143 = add_64 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_72: "f32[16, 512, 1]" = var_mean_18[0]
        getitem_73: "f32[16, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_67: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_18: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_19: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_73);  add_66 = getitem_73 = None
        mul_64: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_65: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg149_1);  mul_64 = arg149_1 = None
        add_68: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, arg150_1);  mul_65 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_144: "f32[8192, 768]" = torch.ops.aten.view.default(add_68, [8192, 768])
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_54: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg152_1, view_144, permute_90);  arg152_1 = view_144 = permute_90 = None
        view_145: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_54, [16, 512, 768]);  addmm_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_146: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_145, [16, 512, 12, 64]);  view_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_91: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_147: "f32[8192, 768]" = torch.ops.aten.view.default(add_68, [8192, 768])
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_55: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg154_1, view_147, permute_92);  arg154_1 = view_147 = permute_92 = None
        view_148: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_55, [16, 512, 768]);  addmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_149: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_148, [16, 512, 12, 64]);  view_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_93: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_150: "f32[8192, 768]" = torch.ops.aten.view.default(add_68, [8192, 768])
        permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_56: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg156_1, view_150, permute_94);  arg156_1 = view_150 = permute_94 = None
        view_151: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_56, [16, 512, 768]);  addmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_152: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_151, [16, 512, 12, 64]);  view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_95: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_11: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_91, permute_93, permute_95, expand_11, False);  permute_91 = permute_93 = permute_95 = expand_11 = None
        getitem_74: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_96: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_153: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_96, [16, 512, 768]);  permute_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_154: "f32[8192, 768]" = torch.ops.aten.view.default(view_153, [8192, 768]);  view_153 = None
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_57: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg158_1, view_154, permute_97);  arg158_1 = view_154 = permute_97 = None
        view_155: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_57, [16, 512, 768]);  addmm_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_69: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_155, add_68);  view_155 = add_68 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_78: "f32[16, 512, 1]" = var_mean_19[0]
        getitem_79: "f32[16, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_70: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_19: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_20: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_79);  add_69 = getitem_79 = None
        mul_66: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_67: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg159_1);  mul_66 = arg159_1 = None
        add_71: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, arg160_1);  mul_67 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_156: "f32[8192, 768]" = torch.ops.aten.view.default(add_71, [8192, 768])
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_58: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg162_1, view_156, permute_98);  arg162_1 = view_156 = permute_98 = None
        view_157: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_58, [16, 512, 3072]);  addmm_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_68: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_69: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_72: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_158: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_70, [8192, 3072]);  mul_70 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_59: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg164_1, view_158, permute_99);  arg164_1 = view_158 = permute_99 = None
        view_159: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_59, [16, 512, 768]);  addmm_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_73: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_159, add_71);  view_159 = add_71 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_80: "f32[16, 512, 1]" = var_mean_20[0]
        getitem_81: "f32[16, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_74: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_20: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_21: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_81);  add_73 = getitem_81 = None
        mul_71: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_72: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, arg165_1);  mul_71 = arg165_1 = None
        add_75: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_72, arg166_1);  mul_72 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_160: "f32[8192, 768]" = torch.ops.aten.view.default(add_75, [8192, 768])
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_60: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg168_1, view_160, permute_100);  arg168_1 = view_160 = permute_100 = None
        view_161: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_60, [16, 512, 768]);  addmm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_162: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_161, [16, 512, 12, 64]);  view_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_101: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_163: "f32[8192, 768]" = torch.ops.aten.view.default(add_75, [8192, 768])
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_61: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg170_1, view_163, permute_102);  arg170_1 = view_163 = permute_102 = None
        view_164: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_61, [16, 512, 768]);  addmm_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_165: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_164, [16, 512, 12, 64]);  view_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_103: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_166: "f32[8192, 768]" = torch.ops.aten.view.default(add_75, [8192, 768])
        permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_62: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg172_1, view_166, permute_104);  arg172_1 = view_166 = permute_104 = None
        view_167: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_62, [16, 512, 768]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_168: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_167, [16, 512, 12, 64]);  view_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_105: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_12: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_101, permute_103, permute_105, expand_12, False);  permute_101 = permute_103 = permute_105 = expand_12 = None
        getitem_82: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_169: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_106, [16, 512, 768]);  permute_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[8192, 768]" = torch.ops.aten.view.default(view_169, [8192, 768]);  view_169 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_63: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg174_1, view_170, permute_107);  arg174_1 = view_170 = permute_107 = None
        view_171: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_63, [16, 512, 768]);  addmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_76: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_171, add_75);  view_171 = add_75 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_86: "f32[16, 512, 1]" = var_mean_21[0]
        getitem_87: "f32[16, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_77: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_21: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_22: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_87);  add_76 = getitem_87 = None
        mul_73: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_74: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg175_1);  mul_73 = arg175_1 = None
        add_78: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_74, arg176_1);  mul_74 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[8192, 768]" = torch.ops.aten.view.default(add_78, [8192, 768])
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_64: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg178_1, view_172, permute_108);  arg178_1 = view_172 = permute_108 = None
        view_173: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_64, [16, 512, 3072]);  addmm_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_75: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_76: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_79: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_79);  mul_75 = add_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_77, [8192, 3072]);  mul_77 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_65: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg180_1, view_174, permute_109);  arg180_1 = view_174 = permute_109 = None
        view_175: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_65, [16, 512, 768]);  addmm_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_80: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_175, add_78);  view_175 = add_78 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_88: "f32[16, 512, 1]" = var_mean_22[0]
        getitem_89: "f32[16, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_81: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_22: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_23: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_89);  add_80 = getitem_89 = None
        mul_78: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_79: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg181_1);  mul_78 = arg181_1 = None
        add_82: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, arg182_1);  mul_79 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:325 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_176: "f32[8192, 768]" = torch.ops.aten.view.default(add_82, [8192, 768])
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_66: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg184_1, view_176, permute_110);  arg184_1 = view_176 = permute_110 = None
        view_177: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_66, [16, 512, 768]);  addmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_178: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_177, [16, 512, 12, 64]);  view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_111: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:338 in forward, code: key_layer = self.transpose_for_scores(self.key(current_states))
        view_179: "f32[8192, 768]" = torch.ops.aten.view.default(add_82, [8192, 768])
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_67: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg186_1, view_179, permute_112);  arg186_1 = view_179 = permute_112 = None
        view_180: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_67, [16, 512, 768]);  addmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_181: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_180, [16, 512, 12, 64]);  view_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_113: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:339 in forward, code: value_layer = self.transpose_for_scores(self.value(current_states))
        view_182: "f32[8192, 768]" = torch.ops.aten.view.default(add_82, [8192, 768])
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_68: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg188_1, view_182, permute_114);  arg188_1 = view_182 = permute_114 = None
        view_183: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_68, [16, 512, 768]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:181 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_184: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(view_183, [16, 512, 12, 64]);  view_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:182 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_115: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:370 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_13: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(where, [16, 12, 512, 512]);  where = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_111, permute_113, permute_115, expand_13, False);  permute_111 = permute_113 = permute_115 = expand_13 = None
        getitem_90: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_116: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_185: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_116, [16, 512, 768]);  permute_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:397 in forward, code: hidden_states = self.dense(hidden_states)
        view_186: "f32[8192, 768]" = torch.ops.aten.view.default(view_185, [8192, 768]);  view_185 = None
        permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_69: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg190_1, view_186, permute_117);  arg190_1 = view_186 = permute_117 = None
        view_187: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_69, [16, 512, 768]);  addmm_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:399 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_187, add_82);  view_187 = add_82 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_94: "f32[16, 512, 1]" = var_mean_23[0]
        getitem_95: "f32[16, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_84: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_23: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_24: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_95);  add_83 = getitem_95 = None
        mul_80: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_81: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg191_1);  mul_80 = arg191_1 = None
        add_85: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_81, arg192_1);  mul_81 = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:472 in forward, code: hidden_states = self.dense(hidden_states)
        view_188: "f32[8192, 768]" = torch.ops.aten.view.default(add_85, [8192, 768])
        permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_70: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg194_1, view_188, permute_118);  arg194_1 = view_188 = permute_118 = None
        view_189: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_70, [16, 512, 3072]);  addmm_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_82: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_83: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11: "f32[16, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_86: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_82, add_86);  mul_82 = add_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:486 in forward, code: hidden_states = self.dense(hidden_states)
        view_190: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_84, [8192, 3072]);  mul_84 = None
        permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_71: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg196_1, view_190, permute_119);  arg196_1 = view_190 = permute_119 = None
        view_191: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_71, [16, 512, 768]);  addmm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:488 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_191, add_85);  view_191 = add_85 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_96: "f32[16, 512, 1]" = var_mean_24[0]
        getitem_97: "f32[16, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_88: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_24: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_25: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_97);  add_87 = getitem_97 = None
        mul_85: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_86: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg197_1);  mul_85 = arg197_1 = None
        add_89: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, arg198_1);  mul_86 = arg198_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1655 in forward, code: logits = self.qa_outputs(sequence_output)
        view_192: "f32[8192, 768]" = torch.ops.aten.view.default(add_89, [8192, 768]);  add_89 = None
        permute_120: "f32[768, 2]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_72: "f32[8192, 2]" = torch.ops.aten.addmm.default(arg200_1, view_192, permute_120);  arg200_1 = view_192 = permute_120 = None
        view_193: "f32[16, 512, 2]" = torch.ops.aten.view.default(addmm_72, [16, 512, 2]);  addmm_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1656 in forward, code: start_logits, end_logits = logits.split(1, dim=-1)
        split = torch.ops.aten.split.Tensor(view_193, 1, -1);  view_193 = None
        getitem_98: "f32[16, 512, 1]" = split[0]
        getitem_99: "f32[16, 512, 1]" = split[1];  split = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1657 in forward, code: start_logits = start_logits.squeeze(-1).contiguous()
        squeeze: "f32[16, 512]" = torch.ops.aten.squeeze.dim(getitem_98, -1);  getitem_98 = None
        clone_25: "f32[16, 512]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1658 in forward, code: end_logits = end_logits.squeeze(-1).contiguous()
        squeeze_1: "f32[16, 512]" = torch.ops.aten.squeeze.dim(getitem_99, -1);  getitem_99 = None
        clone_26: "f32[16, 512]" = torch.ops.aten.clone.default(squeeze_1, memory_format = torch.contiguous_format);  squeeze_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1669 in forward, code: start_positions = start_positions.clamp(0, ignored_index)
        clamp_min: "i64[16]" = torch.ops.aten.clamp_min.default(arg201_1, 0);  arg201_1 = None
        clamp_max: "i64[16]" = torch.ops.aten.clamp_max.default(clamp_min, 512);  clamp_min = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1670 in forward, code: end_positions = end_positions.clamp(0, ignored_index)
        clamp_min_1: "i64[16]" = torch.ops.aten.clamp_min.default(arg202_1, 0);  arg202_1 = None
        clamp_max_1: "i64[16]" = torch.ops.aten.clamp_max.default(clamp_min_1, 512);  clamp_min_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1673 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        amax: "f32[16, 1]" = torch.ops.aten.amax.default(clone_25, [1], True)
        sub_26: "f32[16, 512]" = torch.ops.aten.sub.Tensor(clone_25, amax);  amax = None
        exp: "f32[16, 512]" = torch.ops.aten.exp.default(sub_26)
        sum_1: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27: "f32[16, 512]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne_1: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[16]" = torch.ops.aten.where.self(ne_1, clamp_max, full_default);  ne_1 = full_default = None
        unsqueeze_2: "i64[16, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[16, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_2);  sub_27 = unsqueeze_2 = None
        squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        ne_2: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max, 512)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[16]" = torch.ops.aten.where.self(ne_2, neg, full_default_1);  ne_2 = neg = full_default_1 = None
        ne_3: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max, 512);  clamp_max = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_4: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_4);  sum_3 = convert_element_type_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1674 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        amax_1: "f32[16, 1]" = torch.ops.aten.amax.default(clone_26, [1], True)
        sub_28: "f32[16, 512]" = torch.ops.aten.sub.Tensor(clone_26, amax_1);  amax_1 = None
        exp_1: "f32[16, 512]" = torch.ops.aten.exp.default(sub_28)
        sum_4: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1: "f32[16, 1]" = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_29: "f32[16, 512]" = torch.ops.aten.sub.Tensor(sub_28, log_1);  sub_28 = log_1 = None
        ne_4: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[16]" = torch.ops.aten.where.self(ne_4, clamp_max_1, full_default_2);  ne_4 = full_default_2 = None
        unsqueeze_3: "i64[16, 1]" = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather_1: "f32[16, 1]" = torch.ops.aten.gather.default(sub_29, 1, unsqueeze_3);  sub_29 = unsqueeze_3 = None
        squeeze_3: "f32[16]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1: "f32[16]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        ne_5: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[16]" = torch.ops.aten.where.self(ne_5, neg_1, full_default_3);  ne_5 = neg_1 = full_default_3 = None
        ne_6: "b8[16]" = torch.ops.aten.ne.Scalar(clamp_max_1, 512);  clamp_max_1 = None
        sum_5: "i64[]" = torch.ops.aten.sum.default(ne_6);  ne_6 = None
        convert_element_type_5: "f32[]" = torch.ops.prims.convert_element_type.default(sum_5, torch.float32);  sum_5 = None
        sum_6: "f32[]" = torch.ops.aten.sum.default(where_4);  where_4 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_6, convert_element_type_5);  sum_6 = convert_element_type_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1675 in forward, code: total_loss = (start_loss + end_loss) / 2
        add_90: "f32[]" = torch.ops.aten.add.Tensor(div, div_1);  div = div_1 = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(add_90, 2);  add_90 = None
        return (div_2, clone_25, clone_26)
        