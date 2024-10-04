class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg3_1: "f32[30000, 128]", arg4_1: "f32[2, 128]", arg5_1: "f32[512, 128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[4096, 128]", arg9_1: "f32[4096]", arg10_1: "f32[4096, 4096]", arg11_1: "f32[4096]", arg12_1: "f32[4096, 4096]", arg13_1: "f32[4096]", arg14_1: "f32[4096, 4096]", arg15_1: "f32[4096]", arg16_1: "f32[4096, 4096]", arg17_1: "f32[4096]", arg18_1: "f32[4096]", arg19_1: "f32[4096]", arg20_1: "f32[16384, 4096]", arg21_1: "f32[16384]", arg22_1: "f32[4096, 16384]", arg23_1: "f32[4096]", arg24_1: "f32[4096]", arg25_1: "f32[4096]", arg26_1: "f32[128, 4096]", arg27_1: "f32[128]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[30000]", arg31_1: "i64[4, 512]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:244 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[4, 512, 128]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:777 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[4, 512]" = torch.ops.aten.expand.default(arg1_1, [4, 512]);  arg1_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:245 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[4, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:247 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:249 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, arg2_1);  arg5_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:250 in forward, code: embeddings += position_embeddings
        add_1: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:251 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 512, 1]" = var_mean[0]
        getitem_1: "f32[4, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        add_2: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:521 in forward, code: hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        view: "f32[2048, 128]" = torch.ops.aten.reshape.default(add_3, [2048, 128]);  add_3 = None
        permute: "f32[128, 4096]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm, [4, 512, 4096]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_2: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_1, [2048, 4096])
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_1: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_2, permute_1);  view_2 = permute_1 = None
        view_3: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_1, [4, 512, 4096]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_4: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_3, [4, 512, 64, 64]);  view_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_2: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_5: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_1, [2048, 4096])
        permute_3: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_2: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_5, permute_3);  view_5 = permute_3 = None
        view_6: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 4096]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_7: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_6, [4, 512, 64, 64]);  view_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_4: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_8: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_1, [2048, 4096])
        permute_5: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_3: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_8, permute_5);  view_8 = permute_5 = None
        view_9: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_3, [4, 512, 4096]);  addmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_10: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_9, [4, 512, 64, 64]);  view_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_6: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:773 in forward, code: attention_mask = torch.ones(input_shape, device=device)
        full: "f32[4, 512]" = torch.ops.aten.full.default([4, 512], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:184 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze: "f32[4, 1, 512]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1: "f32[4, 1, 1, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1: "f32[4, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_1, [4, 1, 512, 512]);  unsqueeze_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:186 in _expand_mask, code: inverted_mask = 1.0 - expanded_mask
        sub_1: "f32[4, 1, 512, 512]" = torch.ops.aten.sub.Tensor(1.0, expand_1);  expand_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:188 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type: "b8[4, 1, 512, 512]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(convert_element_type, scalar_tensor, sub_1);  convert_element_type = scalar_tensor = sub_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute_4, permute_6, expand_2, False);  permute_2 = permute_4 = permute_6 = expand_2 = None
        getitem_2: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_7: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_11: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_7, [4, 512, 4096]);  permute_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_12: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_11, [2048, 4096]);  view_11 = None
        permute_8: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_12, permute_8);  view_12 = permute_8 = None
        add_tensor_36: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_36, arg17_1);  mm_default_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_13: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_36, [4, 512, 4096]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_4: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_1, view_13);  view_1 = view_13 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6: "f32[4, 512, 1]" = var_mean_1[0]
        getitem_7: "f32[4, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        add_5: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_2: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_2, arg18_1);  mul_2 = None
        add_6: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_3, arg19_1);  mul_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_14: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_6, [2048, 4096])
        permute_9: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_14, permute_9);  view_14 = permute_9 = None
        add_tensor_35: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_35, arg21_1);  mm_default_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_15: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_35, [4, 512, 16384]);  add_tensor_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
        pow_1: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_15, 3.0)
        mul_5: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_15, mul_5);  view_15 = mul_5 = None
        mul_6: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_16: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_7, [2048, 16384]);  mul_7 = None
        permute_10: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_16, permute_10);  view_16 = permute_10 = None
        add_tensor_34: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_34, arg23_1);  mm_default_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_17: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_34, [4, 512, 4096]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_9: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_17, add_6);  view_17 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_8: "f32[4, 512, 1]" = var_mean_2[0]
        getitem_9: "f32[4, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_9, getitem_9);  add_9 = getitem_9 = None
        add_10: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_8: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_9: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_8, arg24_1);  mul_8 = None
        add_11: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_9, arg25_1);  mul_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_18: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_11, [2048, 4096])
        permute_11: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_7: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_18, permute_11);  view_18 = permute_11 = None
        view_19: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_7, [4, 512, 4096]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_20: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_19, [4, 512, 64, 64]);  view_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_12: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_21: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_11, [2048, 4096])
        permute_13: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_8: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_21, permute_13);  view_21 = permute_13 = None
        view_22: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_8, [4, 512, 4096]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_23: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_22, [4, 512, 64, 64]);  view_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_14: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_24: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_11, [2048, 4096])
        permute_15: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_9: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_24, permute_15);  view_24 = permute_15 = None
        view_25: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_9, [4, 512, 4096]);  addmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_26: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_25, [4, 512, 64, 64]);  view_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_16: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_3: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_12, permute_14, permute_16, expand_3, False);  permute_12 = permute_14 = permute_16 = expand_3 = None
        getitem_10: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_17: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_27: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_17, [4, 512, 4096]);  permute_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_28: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_27, [2048, 4096]);  view_27 = None
        permute_18: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_28, permute_18);  view_28 = permute_18 = None
        add_tensor_33: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_33, arg17_1);  mm_default_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_29: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_33, [4, 512, 4096]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_12: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_11, view_29);  add_11 = view_29 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_14: "f32[4, 512, 1]" = var_mean_3[0]
        getitem_15: "f32[4, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_4: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_12, getitem_15);  add_12 = getitem_15 = None
        add_13: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_10: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_11: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_10, arg18_1);  mul_10 = None
        add_14: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_11, arg19_1);  mul_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_30: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_14, [2048, 4096])
        permute_19: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_30, permute_19);  view_30 = permute_19 = None
        add_tensor_32: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_32, arg21_1);  mm_default_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_31: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_32, [4, 512, 16384]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
        pow_2: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_31, 3.0)
        mul_13: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_31, mul_13);  view_31 = mul_13 = None
        mul_14: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_32: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_15, [2048, 16384]);  mul_15 = None
        permute_20: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_32, permute_20);  view_32 = permute_20 = None
        add_tensor_31: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_31, arg23_1);  mm_default_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_33: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_31, [4, 512, 4096]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_17: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_33, add_14);  view_33 = add_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_16: "f32[4, 512, 1]" = var_mean_4[0]
        getitem_17: "f32[4, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_5: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_17, getitem_17);  add_17 = getitem_17 = None
        add_18: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_16: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_17: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_16, arg24_1);  mul_16 = None
        add_19: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_17, arg25_1);  mul_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_34: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_19, [2048, 4096])
        permute_21: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_13: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_34, permute_21);  view_34 = permute_21 = None
        view_35: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_13, [4, 512, 4096]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_36: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_35, [4, 512, 64, 64]);  view_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_22: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_37: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_19, [2048, 4096])
        permute_23: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_14: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_37, permute_23);  view_37 = permute_23 = None
        view_38: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_14, [4, 512, 4096]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_39: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_38, [4, 512, 64, 64]);  view_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_24: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_40: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_19, [2048, 4096])
        permute_25: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_15: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_40, permute_25);  view_40 = permute_25 = None
        view_41: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_15, [4, 512, 4096]);  addmm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_42: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_41, [4, 512, 64, 64]);  view_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_26: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_24, permute_26, expand_4, False);  permute_22 = permute_24 = permute_26 = expand_4 = None
        getitem_18: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_27: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_43: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_27, [4, 512, 4096]);  permute_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_44: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_43, [2048, 4096]);  view_43 = None
        permute_28: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_44, permute_28);  view_44 = permute_28 = None
        add_tensor_30: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_30, arg17_1);  mm_default_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_45: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_30, [4, 512, 4096]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_20: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_19, view_45);  add_19 = view_45 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_22: "f32[4, 512, 1]" = var_mean_5[0]
        getitem_23: "f32[4, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_6: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_20, getitem_23);  add_20 = getitem_23 = None
        add_21: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_18: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_19: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_18, arg18_1);  mul_18 = None
        add_22: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_19, arg19_1);  mul_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_46: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_22, [2048, 4096])
        permute_29: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_46, permute_29);  view_46 = permute_29 = None
        add_tensor_29: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_29, arg21_1);  mm_default_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_47: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_29, [4, 512, 16384]);  add_tensor_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
        pow_3: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
        mul_21: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_47, mul_21);  view_47 = mul_21 = None
        mul_22: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_48: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_23, [2048, 16384]);  mul_23 = None
        permute_30: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_48, permute_30);  view_48 = permute_30 = None
        add_tensor_28: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_28, arg23_1);  mm_default_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_49: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_28, [4, 512, 4096]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_25: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_49, add_22);  view_49 = add_22 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_24: "f32[4, 512, 1]" = var_mean_6[0]
        getitem_25: "f32[4, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_7: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_25, getitem_25);  add_25 = getitem_25 = None
        add_26: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_24: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_25: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_24, arg24_1);  mul_24 = None
        add_27: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_25, arg25_1);  mul_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_50: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_27, [2048, 4096])
        permute_31: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_19: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_50, permute_31);  view_50 = permute_31 = None
        view_51: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_19, [4, 512, 4096]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_52: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_51, [4, 512, 64, 64]);  view_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_32: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_53: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_27, [2048, 4096])
        permute_33: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_20: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_53, permute_33);  view_53 = permute_33 = None
        view_54: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_20, [4, 512, 4096]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_55: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_54, [4, 512, 64, 64]);  view_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_34: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_56: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_27, [2048, 4096])
        permute_35: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_21: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_56, permute_35);  view_56 = permute_35 = None
        view_57: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_21, [4, 512, 4096]);  addmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_58: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_57, [4, 512, 64, 64]);  view_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_36: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_5: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_32, permute_34, permute_36, expand_5, False);  permute_32 = permute_34 = permute_36 = expand_5 = None
        getitem_26: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_37: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_59: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_37, [4, 512, 4096]);  permute_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_60: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_59, [2048, 4096]);  view_59 = None
        permute_38: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_60, permute_38);  view_60 = permute_38 = None
        add_tensor_27: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_27, arg17_1);  mm_default_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_61: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_27, [4, 512, 4096]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_28: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_27, view_61);  add_27 = view_61 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_30: "f32[4, 512, 1]" = var_mean_7[0]
        getitem_31: "f32[4, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_8: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_28, getitem_31);  add_28 = getitem_31 = None
        add_29: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_26: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_27: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_26, arg18_1);  mul_26 = None
        add_30: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_27, arg19_1);  mul_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_62: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_30, [2048, 4096])
        permute_39: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_62, permute_39);  view_62 = permute_39 = None
        add_tensor_26: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_26, arg21_1);  mm_default_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_63: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_26, [4, 512, 16384]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        pow_4: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_63, 3.0)
        mul_29: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_63, mul_29);  view_63 = mul_29 = None
        mul_30: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_64: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_31, [2048, 16384]);  mul_31 = None
        permute_40: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_64, permute_40);  view_64 = permute_40 = None
        add_tensor_25: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_25, arg23_1);  mm_default_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_65: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_25, [4, 512, 4096]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_33: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_65, add_30);  view_65 = add_30 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_32: "f32[4, 512, 1]" = var_mean_8[0]
        getitem_33: "f32[4, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_9: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_33, getitem_33);  add_33 = getitem_33 = None
        add_34: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_32: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_33: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_32, arg24_1);  mul_32 = None
        add_35: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_33, arg25_1);  mul_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_66: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_35, [2048, 4096])
        permute_41: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_25: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_66, permute_41);  view_66 = permute_41 = None
        view_67: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_25, [4, 512, 4096]);  addmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_68: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_67, [4, 512, 64, 64]);  view_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_42: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_69: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_35, [2048, 4096])
        permute_43: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_26: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_69, permute_43);  view_69 = permute_43 = None
        view_70: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_26, [4, 512, 4096]);  addmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_71: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_70, [4, 512, 64, 64]);  view_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_44: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_72: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_35, [2048, 4096])
        permute_45: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_27: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_72, permute_45);  view_72 = permute_45 = None
        view_73: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_27, [4, 512, 4096]);  addmm_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_74: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_73, [4, 512, 64, 64]);  view_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_46: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_44, permute_46, expand_6, False);  permute_42 = permute_44 = permute_46 = expand_6 = None
        getitem_34: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_47: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_75: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_47, [4, 512, 4096]);  permute_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_76: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_75, [2048, 4096]);  view_75 = None
        permute_48: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_76, permute_48);  view_76 = permute_48 = None
        add_tensor_24: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_24, arg17_1);  mm_default_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_77: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_24, [4, 512, 4096]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_36: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_35, view_77);  add_35 = view_77 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_38: "f32[4, 512, 1]" = var_mean_9[0]
        getitem_39: "f32[4, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_10: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_36, getitem_39);  add_36 = getitem_39 = None
        add_37: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_34: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_35: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_34, arg18_1);  mul_34 = None
        add_38: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_35, arg19_1);  mul_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_78: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_38, [2048, 4096])
        permute_49: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_78, permute_49);  view_78 = permute_49 = None
        add_tensor_23: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_23, arg21_1);  mm_default_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_79: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_23, [4, 512, 16384]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_79, 0.5)
        pow_5: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_79, 3.0)
        mul_37: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_79, mul_37);  view_79 = mul_37 = None
        mul_38: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_80: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_39, [2048, 16384]);  mul_39 = None
        permute_50: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_80, permute_50);  view_80 = permute_50 = None
        add_tensor_22: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_22, arg23_1);  mm_default_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_81: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_22, [4, 512, 4096]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_41: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_81, add_38);  view_81 = add_38 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_40: "f32[4, 512, 1]" = var_mean_10[0]
        getitem_41: "f32[4, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_11: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_41, getitem_41);  add_41 = getitem_41 = None
        add_42: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_40: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_41: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, arg24_1);  mul_40 = None
        add_43: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_41, arg25_1);  mul_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_82: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_43, [2048, 4096])
        permute_51: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_31: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_82, permute_51);  view_82 = permute_51 = None
        view_83: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_31, [4, 512, 4096]);  addmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_84: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_83, [4, 512, 64, 64]);  view_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_52: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_85: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_43, [2048, 4096])
        permute_53: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_32: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_85, permute_53);  view_85 = permute_53 = None
        view_86: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_32, [4, 512, 4096]);  addmm_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_87: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_86, [4, 512, 64, 64]);  view_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_54: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_88: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_43, [2048, 4096])
        permute_55: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_33: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_88, permute_55);  view_88 = permute_55 = None
        view_89: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_33, [4, 512, 4096]);  addmm_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_90: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_89, [4, 512, 64, 64]);  view_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_56: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_7: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_52, permute_54, permute_56, expand_7, False);  permute_52 = permute_54 = permute_56 = expand_7 = None
        getitem_42: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_57: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_91: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_57, [4, 512, 4096]);  permute_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_92: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_91, [2048, 4096]);  view_91 = None
        permute_58: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_92, permute_58);  view_92 = permute_58 = None
        add_tensor_21: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_21, arg17_1);  mm_default_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_93: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_21, [4, 512, 4096]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_44: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_43, view_93);  add_43 = view_93 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_46: "f32[4, 512, 1]" = var_mean_11[0]
        getitem_47: "f32[4, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_12: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_44, getitem_47);  add_44 = getitem_47 = None
        add_45: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_42: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_43: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_42, arg18_1);  mul_42 = None
        add_46: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_43, arg19_1);  mul_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_94: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_46, [2048, 4096])
        permute_59: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_94, permute_59);  view_94 = permute_59 = None
        add_tensor_20: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_20, arg21_1);  mm_default_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_95: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_20, [4, 512, 16384]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_95, 0.5)
        pow_6: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_95, 3.0)
        mul_45: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_95, mul_45);  view_95 = mul_45 = None
        mul_46: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_96: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_47, [2048, 16384]);  mul_47 = None
        permute_60: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_96, permute_60);  view_96 = permute_60 = None
        add_tensor_19: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_19, arg23_1);  mm_default_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_97: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_19, [4, 512, 4096]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_49: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_97, add_46);  view_97 = add_46 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_48: "f32[4, 512, 1]" = var_mean_12[0]
        getitem_49: "f32[4, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_13: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_49, getitem_49);  add_49 = getitem_49 = None
        add_50: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_48: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_49: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_48, arg24_1);  mul_48 = None
        add_51: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_49, arg25_1);  mul_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_98: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_51, [2048, 4096])
        permute_61: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_37: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_98, permute_61);  view_98 = permute_61 = None
        view_99: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_37, [4, 512, 4096]);  addmm_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_100: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_99, [4, 512, 64, 64]);  view_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_62: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_101: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_51, [2048, 4096])
        permute_63: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_38: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_101, permute_63);  view_101 = permute_63 = None
        view_102: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_38, [4, 512, 4096]);  addmm_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_103: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_102, [4, 512, 64, 64]);  view_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_64: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_104: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_51, [2048, 4096])
        permute_65: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_39: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_104, permute_65);  view_104 = permute_65 = None
        view_105: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_39, [4, 512, 4096]);  addmm_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_106: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_105, [4, 512, 64, 64]);  view_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_66: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_62, permute_64, permute_66, expand_8, False);  permute_62 = permute_64 = permute_66 = expand_8 = None
        getitem_50: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_67: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_107: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_67, [4, 512, 4096]);  permute_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_108: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_107, [2048, 4096]);  view_107 = None
        permute_68: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_108, permute_68);  view_108 = permute_68 = None
        add_tensor_18: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_18, arg17_1);  mm_default_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_109: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_18, [4, 512, 4096]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_52: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_51, view_109);  add_51 = view_109 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_54: "f32[4, 512, 1]" = var_mean_13[0]
        getitem_55: "f32[4, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_14: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_52, getitem_55);  add_52 = getitem_55 = None
        add_53: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_13: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_50: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_51: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_50, arg18_1);  mul_50 = None
        add_54: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_51, arg19_1);  mul_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_110: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_54, [2048, 4096])
        permute_69: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_110, permute_69);  view_110 = permute_69 = None
        add_tensor_17: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_17, arg21_1);  mm_default_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_111: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_17, [4, 512, 16384]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
        pow_7: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_111, 3.0)
        mul_53: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_111, mul_53);  view_111 = mul_53 = None
        mul_54: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_56: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_112: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_55, [2048, 16384]);  mul_55 = None
        permute_70: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_112, permute_70);  view_112 = permute_70 = None
        add_tensor_16: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_16, arg23_1);  mm_default_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_113: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_16, [4, 512, 4096]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_57: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_113, add_54);  view_113 = add_54 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_56: "f32[4, 512, 1]" = var_mean_14[0]
        getitem_57: "f32[4, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_15: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_57, getitem_57);  add_57 = getitem_57 = None
        add_58: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_14: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_56: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_57: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_56, arg24_1);  mul_56 = None
        add_59: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_57, arg25_1);  mul_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_114: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_59, [2048, 4096])
        permute_71: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_43: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_114, permute_71);  view_114 = permute_71 = None
        view_115: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_43, [4, 512, 4096]);  addmm_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_116: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_115, [4, 512, 64, 64]);  view_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_72: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_117: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_59, [2048, 4096])
        permute_73: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_44: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_117, permute_73);  view_117 = permute_73 = None
        view_118: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_44, [4, 512, 4096]);  addmm_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_119: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_118, [4, 512, 64, 64]);  view_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_74: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_120: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_59, [2048, 4096])
        permute_75: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_45: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_120, permute_75);  view_120 = permute_75 = None
        view_121: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_45, [4, 512, 4096]);  addmm_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_122: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_121, [4, 512, 64, 64]);  view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_76: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_9: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_72, permute_74, permute_76, expand_9, False);  permute_72 = permute_74 = permute_76 = expand_9 = None
        getitem_58: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_77: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_123: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_77, [4, 512, 4096]);  permute_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_124: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_123, [2048, 4096]);  view_123 = None
        permute_78: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_124, permute_78);  view_124 = permute_78 = None
        add_tensor_15: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_15, arg17_1);  mm_default_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_125: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_15, [4, 512, 4096]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_60: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_59, view_125);  add_59 = view_125 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_62: "f32[4, 512, 1]" = var_mean_15[0]
        getitem_63: "f32[4, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_16: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_60, getitem_63);  add_60 = getitem_63 = None
        add_61: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_15: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_58: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_59: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_58, arg18_1);  mul_58 = None
        add_62: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_59, arg19_1);  mul_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_126: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_62, [2048, 4096])
        permute_79: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_126, permute_79);  view_126 = permute_79 = None
        add_tensor_14: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_14, arg21_1);  mm_default_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_127: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_14, [4, 512, 16384]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_127, 0.5)
        pow_8: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_127, 3.0)
        mul_61: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_127, mul_61);  view_127 = mul_61 = None
        mul_62: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_64: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_128: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_63, [2048, 16384]);  mul_63 = None
        permute_80: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_128, permute_80);  view_128 = permute_80 = None
        add_tensor_13: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_13, arg23_1);  mm_default_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_129: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_13, [4, 512, 4096]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_65: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_129, add_62);  view_129 = add_62 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_64: "f32[4, 512, 1]" = var_mean_16[0]
        getitem_65: "f32[4, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_17: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_65, getitem_65);  add_65 = getitem_65 = None
        add_66: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_16: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_64: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_65: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_64, arg24_1);  mul_64 = None
        add_67: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_65, arg25_1);  mul_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_130: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_67, [2048, 4096])
        permute_81: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_49: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_130, permute_81);  view_130 = permute_81 = None
        view_131: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_49, [4, 512, 4096]);  addmm_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_132: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_131, [4, 512, 64, 64]);  view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_82: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_133: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_67, [2048, 4096])
        permute_83: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_50: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_133, permute_83);  view_133 = permute_83 = None
        view_134: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_50, [4, 512, 4096]);  addmm_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_135: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_134, [4, 512, 64, 64]);  view_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_84: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_136: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_67, [2048, 4096])
        permute_85: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_51: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_136, permute_85);  view_136 = permute_85 = None
        view_137: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_51, [4, 512, 4096]);  addmm_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_138: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_137, [4, 512, 64, 64]);  view_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_86: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_82, permute_84, permute_86, expand_10, False);  permute_82 = permute_84 = permute_86 = expand_10 = None
        getitem_66: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_87: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_139: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_87, [4, 512, 4096]);  permute_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_140: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_139, [2048, 4096]);  view_139 = None
        permute_88: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_140, permute_88);  view_140 = permute_88 = None
        add_tensor_12: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_12, arg17_1);  mm_default_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_141: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_12, [4, 512, 4096]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_68: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_67, view_141);  add_67 = view_141 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_70: "f32[4, 512, 1]" = var_mean_17[0]
        getitem_71: "f32[4, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_18: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_68, getitem_71);  add_68 = getitem_71 = None
        add_69: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_17: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_66: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_67: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_66, arg18_1);  mul_66 = None
        add_70: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_67, arg19_1);  mul_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_142: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_70, [2048, 4096])
        permute_89: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_142, permute_89);  view_142 = permute_89 = None
        add_tensor_11: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_11, arg21_1);  mm_default_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_143: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_11, [4, 512, 16384]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
        pow_9: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_143, 3.0)
        mul_69: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_143, mul_69);  view_143 = mul_69 = None
        mul_70: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_72: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_144: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_71, [2048, 16384]);  mul_71 = None
        permute_90: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_144, permute_90);  view_144 = permute_90 = None
        add_tensor_10: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_10, arg23_1);  mm_default_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_145: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_10, [4, 512, 4096]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_73: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_145, add_70);  view_145 = add_70 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_72: "f32[4, 512, 1]" = var_mean_18[0]
        getitem_73: "f32[4, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_19: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_73, getitem_73);  add_73 = getitem_73 = None
        add_74: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_18: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_72: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_73: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_72, arg24_1);  mul_72 = None
        add_75: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_73, arg25_1);  mul_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_146: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_75, [2048, 4096])
        permute_91: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_55: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_146, permute_91);  view_146 = permute_91 = None
        view_147: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_55, [4, 512, 4096]);  addmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_148: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_147, [4, 512, 64, 64]);  view_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_92: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_149: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_75, [2048, 4096])
        permute_93: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_56: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_149, permute_93);  view_149 = permute_93 = None
        view_150: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_56, [4, 512, 4096]);  addmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_151: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_150, [4, 512, 64, 64]);  view_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_94: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_152: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_75, [2048, 4096])
        permute_95: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_57: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_152, permute_95);  view_152 = permute_95 = None
        view_153: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_57, [4, 512, 4096]);  addmm_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_154: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_153, [4, 512, 64, 64]);  view_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_96: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_11: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_92, permute_94, permute_96, expand_11, False);  permute_92 = permute_94 = permute_96 = expand_11 = None
        getitem_74: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_97: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_155: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_97, [4, 512, 4096]);  permute_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_156: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_155, [2048, 4096]);  view_155 = None
        permute_98: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_156, permute_98);  view_156 = permute_98 = None
        add_tensor_9: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_9, arg17_1);  mm_default_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_157: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_9, [4, 512, 4096]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_76: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_75, view_157);  add_75 = view_157 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_78: "f32[4, 512, 1]" = var_mean_19[0]
        getitem_79: "f32[4, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_20: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_76, getitem_79);  add_76 = getitem_79 = None
        add_77: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_19: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_74: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_75: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_74, arg18_1);  mul_74 = None
        add_78: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_75, arg19_1);  mul_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_158: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_78, [2048, 4096])
        permute_99: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_158, permute_99);  view_158 = permute_99 = None
        add_tensor_8: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_8, arg21_1);  mm_default_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_159: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_8, [4, 512, 16384]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
        pow_10: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_159, 3.0)
        mul_77: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_159, mul_77);  view_159 = mul_77 = None
        mul_78: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_80: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_160: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_79, [2048, 16384]);  mul_79 = None
        permute_100: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_160, permute_100);  view_160 = permute_100 = None
        add_tensor_7: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_7, arg23_1);  mm_default_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_161: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_7, [4, 512, 4096]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_81: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_161, add_78);  view_161 = add_78 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_80: "f32[4, 512, 1]" = var_mean_20[0]
        getitem_81: "f32[4, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_21: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_81, getitem_81);  add_81 = getitem_81 = None
        add_82: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_20: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_80: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_81: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_80, arg24_1);  mul_80 = None
        add_83: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_81, arg25_1);  mul_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_162: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_83, [2048, 4096])
        permute_101: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0])
        addmm_61: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_162, permute_101);  view_162 = permute_101 = None
        view_163: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_61, [4, 512, 4096]);  addmm_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_164: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_163, [4, 512, 64, 64]);  view_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_102: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_165: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_83, [2048, 4096])
        permute_103: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0])
        addmm_62: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_165, permute_103);  view_165 = permute_103 = None
        view_166: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_62, [4, 512, 4096]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_167: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_166, [4, 512, 64, 64]);  view_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_104: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_168: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_83, [2048, 4096])
        permute_105: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0])
        addmm_63: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_168, permute_105);  view_168 = permute_105 = None
        view_169: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_63, [4, 512, 4096]);  addmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_170: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_169, [4, 512, 64, 64]);  view_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_106: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_12: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_102, permute_104, permute_106, expand_12, False);  permute_102 = permute_104 = permute_106 = expand_12 = None
        getitem_82: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_107: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_171: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_107, [4, 512, 4096]);  permute_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_172: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_171, [2048, 4096]);  view_171 = None
        permute_108: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_172, permute_108);  view_172 = permute_108 = None
        add_tensor_6: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_6, arg17_1);  mm_default_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_173: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_6, [4, 512, 4096]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_84: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_83, view_173);  add_83 = view_173 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_86: "f32[4, 512, 1]" = var_mean_21[0]
        getitem_87: "f32[4, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_22: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_84, getitem_87);  add_84 = getitem_87 = None
        add_85: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_21: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_82: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_83: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_82, arg18_1);  mul_82 = None
        add_86: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_83, arg19_1);  mul_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_174: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_86, [2048, 4096])
        permute_109: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_174, permute_109);  view_174 = permute_109 = None
        add_tensor_5: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_5, arg21_1);  mm_default_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_175: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_5, [4, 512, 16384]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        pow_11: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
        mul_85: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_175, mul_85);  view_175 = mul_85 = None
        mul_86: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_88: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_176: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_87, [2048, 16384]);  mul_87 = None
        permute_110: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0])
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_176, permute_110);  view_176 = permute_110 = None
        add_tensor_4: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_4, arg23_1);  mm_default_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_177: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_4, [4, 512, 4096]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_89: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_177, add_86);  view_177 = add_86 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_88: "f32[4, 512, 1]" = var_mean_22[0]
        getitem_89: "f32[4, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_23: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_89, getitem_89);  add_89 = getitem_89 = None
        add_90: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_22: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_88: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_89: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_88, arg24_1);  mul_88 = None
        add_91: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_89, arg25_1);  mul_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:392 in forward, code: query_layer = self.transpose_for_scores(self.query(hidden_states))
        view_178: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_91, [2048, 4096])
        permute_111: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_67: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg11_1, view_178, permute_111);  arg11_1 = view_178 = permute_111 = None
        view_179: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_67, [4, 512, 4096]);  addmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_180: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_179, [4, 512, 64, 64]);  view_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_112: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:393 in forward, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
        view_181: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_91, [2048, 4096])
        permute_113: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_68: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg13_1, view_181, permute_113);  arg13_1 = view_181 = permute_113 = None
        view_182: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_68, [4, 512, 4096]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_183: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_182, [4, 512, 64, 64]);  view_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_114: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:394 in forward, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
        view_184: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_91, [2048, 4096])
        permute_115: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_69: "f32[2048, 4096]" = torch.ops.aten.addmm.default(arg15_1, view_184, permute_115);  arg15_1 = view_184 = permute_115 = None
        view_185: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(addmm_69, [4, 512, 4096]);  addmm_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:288 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_186: "f32[4, 512, 64, 64]" = torch.ops.aten.reshape.default(view_185, [4, 512, 64, 64]);  view_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:289 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_116: "f32[4, 64, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:404 in forward, code: attention_output = torch.nn.functional.scaled_dot_product_attention(
        expand_13: "f32[4, 64, 512, 512]" = torch.ops.aten.expand.default(where, [4, 64, 512, 512]);  where = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_112, permute_114, permute_116, expand_13, False);  permute_112 = permute_114 = permute_116 = expand_13 = None
        getitem_90: "f32[4, 64, 512, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:413 in forward, code: attention_output = attention_output.transpose(1, 2)
        permute_117: "f32[4, 512, 64, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:414 in forward, code: attention_output = attention_output.reshape(batch_size, seq_len, self.all_head_size)
        view_187: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(permute_117, [4, 512, 4096]);  permute_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_188: "f32[2048, 4096]" = torch.ops.aten.reshape.default(view_187, [2048, 4096]);  view_187 = None
        permute_118: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_188, permute_118);  view_188 = permute_118 = None
        add_tensor_3: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_3, arg17_1);  mm_default_3 = arg17_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:416 in forward, code: projected_context_layer = self.dense(attention_output)
        view_189: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_3, [4, 512, 4096]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:418 in forward, code: layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        add_92: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(add_91, view_189);  add_91 = view_189 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_94: "f32[4, 512, 1]" = var_mean_23[0]
        getitem_95: "f32[4, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_24: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_92, getitem_95);  add_92 = getitem_95 = None
        add_93: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_23: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_90: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_91: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_90, arg18_1);  mul_90 = arg18_1 = None
        add_94: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_91, arg19_1);  mul_91 = arg19_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_190: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_94, [2048, 4096])
        permute_119: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[2048, 16384]" = torch.ops.aten.mm.default(view_190, permute_119);  view_190 = permute_119 = None
        add_tensor_2: "f32[2048, 16384]" = torch.ops.aten.add.Tensor(mm_default_2, arg21_1);  mm_default_2 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:463 in ff_chunk, code: ffn_output = self.ffn(attention_output)
        view_191: "f32[4, 512, 16384]" = torch.ops.aten.reshape.default(add_tensor_2, [4, 512, 16384]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(view_191, 0.5)
        pow_12: "f32[4, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_191, 3.0)
        mul_93: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(view_191, mul_93);  view_191 = mul_93 = None
        mul_94: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11: "f32[4, 512, 16384]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_96: "f32[4, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[4, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_192: "f32[2048, 16384]" = torch.ops.aten.reshape.default(mul_95, [2048, 16384]);  mul_95 = None
        permute_120: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[2048, 4096]" = torch.ops.aten.mm.default(view_192, permute_120);  view_192 = permute_120 = None
        add_tensor_1: "f32[2048, 4096]" = torch.ops.aten.add.Tensor(mm_default_1, arg23_1);  mm_default_1 = arg23_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:465 in ff_chunk, code: ffn_output = self.ffn_output(ffn_output)
        view_193: "f32[4, 512, 4096]" = torch.ops.aten.reshape.default(add_tensor_1, [4, 512, 4096]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:458 in forward, code: hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])
        add_97: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(view_193, add_94);  view_193 = add_94 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_96: "f32[4, 512, 1]" = var_mean_24[0]
        getitem_97: "f32[4, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_25: "f32[4, 512, 4096]" = torch.ops.aten.sub.Tensor(add_97, getitem_97);  add_97 = getitem_97 = None
        add_98: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_24: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_96: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_97: "f32[4, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_96, arg24_1);  mul_96 = arg24_1 = None
        add_99: "f32[4, 512, 4096]" = torch.ops.aten.add.Tensor(mul_97, arg25_1);  mul_97 = arg25_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:952 in forward, code: hidden_states = self.dense(hidden_states)
        view_194: "f32[2048, 4096]" = torch.ops.aten.reshape.default(add_99, [2048, 4096]);  add_99 = None
        permute_121: "f32[4096, 128]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[2048, 128]" = torch.ops.aten.mm.default(view_194, permute_121);  view_194 = permute_121 = None
        add_tensor: "f32[2048, 128]" = torch.ops.aten.add.Tensor(mm_default, arg27_1);  mm_default = arg27_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:952 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "f32[4, 512, 128]" = torch.ops.aten.reshape.default(add_tensor, [4, 512, 128]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_98: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        pow_13: "f32[4, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 3.0)
        mul_99: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_100: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(view_195, mul_99);  view_195 = mul_99 = None
        mul_100: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_12: "f32[4, 512, 128]" = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
        add_101: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_101: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_98, add_101);  mul_98 = add_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:954 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_101, [2], correction = 0, keepdim = True)
        getitem_98: "f32[4, 512, 1]" = var_mean_25[0]
        getitem_99: "f32[4, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1081 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_199: "i64[2048]" = torch.ops.aten.reshape.default(arg31_1, [-1]);  arg31_1 = None
        ne_1: "b8[2048]" = torch.ops.aten.ne.Scalar(view_199, -100)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:954 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_26: "f32[4, 512, 128]" = torch.ops.aten.sub.Tensor(mul_101, getitem_99);  mul_101 = getitem_99 = None
        add_102: "f32[4, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
        rsqrt_25: "f32[4, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_102: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_25);  sub_26 = rsqrt_25 = None
        mul_103: "f32[4, 512, 128]" = torch.ops.aten.mul.Tensor(mul_102, arg28_1);  mul_102 = arg28_1 = None
        add_103: "f32[4, 512, 128]" = torch.ops.aten.add.Tensor(mul_103, arg29_1);  mul_103 = arg29_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:955 in forward, code: hidden_states = self.decoder(hidden_states)
        view_196: "f32[2048, 128]" = torch.ops.aten.reshape.default(add_103, [2048, 128]);  add_103 = None
        permute_122: "f32[128, 30000]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_74: "f32[2048, 30000]" = torch.ops.aten.addmm.default(arg30_1, view_196, permute_122);  arg30_1 = view_196 = permute_122 = None
        view_197: "f32[4, 512, 30000]" = torch.ops.aten.reshape.default(addmm_74, [4, 512, 30000]);  addmm_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/albert/modeling_albert.py:1081 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_198: "f32[2048, 30000]" = torch.ops.aten.reshape.default(view_197, [-1, 30000])
        amax: "f32[2048, 1]" = torch.ops.aten.amax.default(view_198, [1], True)
        sub_27: "f32[2048, 30000]" = torch.ops.aten.sub.Tensor(view_198, amax);  view_198 = amax = None
        exp: "f32[2048, 30000]" = torch.ops.aten.exp.default(sub_27)
        sum_1: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_28: "f32[2048, 30000]" = torch.ops.aten.sub.Tensor(sub_27, log);  sub_27 = log = None
        ne: "b8[2048]" = torch.ops.aten.ne.Scalar(view_199, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[2048]" = torch.ops.aten.where.self(ne, view_199, full_default);  ne = full_default = None
        unsqueeze_2: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_28, 1, unsqueeze_2);  sub_28 = unsqueeze_2 = None
        squeeze: "f32[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[2048]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_2: "b8[2048]" = torch.ops.aten.ne.Scalar(view_199, -100);  view_199 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        return (div, view_197)
        