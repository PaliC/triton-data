class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[128, 128]", arg1_1: "f32[30522, 768]", arg2_1: "i64[1, 512]", arg3_1: "f32[512, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768, 768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[3072, 768]", arg17_1: "f32[3072]", arg18_1: "f32[768, 3072]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768, 768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[3072, 768]", arg33_1: "f32[3072]", arg34_1: "f32[768, 3072]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[3072, 768]", arg49_1: "f32[3072]", arg50_1: "f32[768, 3072]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768, 768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[3072, 768]", arg65_1: "f32[3072]", arg66_1: "f32[768, 3072]", arg67_1: "f32[768]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768, 768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[3072, 768]", arg81_1: "f32[3072]", arg82_1: "f32[768, 3072]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[3072, 768]", arg97_1: "f32[3072]", arg98_1: "f32[768, 3072]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768, 768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[30522]", arg107_1: "i64[128, 128]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:116 in forward, code: input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        embedding: "f32[128, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:124 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:129 in forward, code: position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
        embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, slice_2);  arg3_1 = slice_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:131 in forward, code: embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        add: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:132 in forward, code: embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[128, 128, 1]" = var_mean[0]
        getitem_1: "f32[128, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:791 in forward, code: attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)
        full: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:184 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze: "f32[128, 1, 128]" = torch.ops.aten.unsqueeze.default(full, 1);  full = None
        unsqueeze_1: "f32[128, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand: "f32[128, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_1, [128, 1, 128, 128]);  unsqueeze_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:186 in _expand_mask, code: inverted_mask = 1.0 - expanded_mask
        sub_1: "f32[128, 1, 128, 128]" = torch.ops.aten.sub.Tensor(1.0, expand);  expand = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:188 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type: "b8[128, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(-3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where: "f32[128, 1, 128, 128]" = torch.ops.aten.where.self(convert_element_type, scalar_tensor, sub_1);  convert_element_type = scalar_tensor = sub_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, [16384, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm, [128, 128, 768]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_2: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_1, [128, -1, 12, 64]);  view_1 = None
        permute_1: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        view_3: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, [16384, 768])
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg9_1, view_3, permute_2);  arg9_1 = view_3 = permute_2 = None
        view_4: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_1, [128, 128, 768]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_5: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_4, [128, -1, 12, 64]);  view_4 = None
        permute_3: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:393 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        view_6: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, [16384, 768])
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg11_1, view_6, permute_4);  arg11_1 = view_6 = permute_4 = None
        view_7: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_2, [128, 128, 768]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_8: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_7, [128, -1, 12, 64]);  view_7 = None
        permute_5: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:403 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_1: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_1, False);  permute_1 = permute_3 = permute_5 = expand_1 = None
        getitem_2: "f32[128, 12, 128, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:389 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_6: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        view_9: "f32[128, 128, 768]" = torch.ops.aten.view.default(permute_6, [128, -1, 768]);  permute_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:413 in forward, code: attn_output = self.out_lin(attn_output)
        view_10: "f32[16384, 768]" = torch.ops.aten.view.default(view_9, [16384, 768]);  view_9 = None
        permute_7: "f32[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg13_1, view_10, permute_7);  arg13_1 = view_10 = permute_7 = None
        view_11: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_3, [128, 128, 768]);  addmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:492 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_11, add_2);  view_11 = add_2 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_6: "f32[128, 128, 1]" = var_mean_1[0]
        getitem_7: "f32[128, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_7);  add_3 = getitem_7 = None
        mul_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg14_1);  mul_2 = arg14_1 = None
        add_5: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, arg15_1);  mul_3 = arg15_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:432 in ff_chunk, code: x = self.lin1(input)
        view_12: "f32[16384, 768]" = torch.ops.aten.view.default(add_5, [16384, 768])
        permute_8: "f32[768, 3072]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg17_1, view_12, permute_8);  arg17_1 = view_12 = permute_8 = None
        view_13: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [128, 128, 3072]);  addmm_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_4: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:434 in ff_chunk, code: x = self.lin2(x)
        view_14: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_6, [16384, 3072]);  mul_6 = None
        permute_9: "f32[3072, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg19_1, view_14, permute_9);  arg19_1 = view_14 = permute_9 = None
        view_15: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_5, [128, 128, 768]);  addmm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:496 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_7: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_15, add_5);  view_15 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_8: "f32[128, 128, 1]" = var_mean_2[0]
        getitem_9: "f32[128, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_9);  add_7 = getitem_9 = None
        mul_7: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_8: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg20_1);  mul_7 = arg20_1 = None
        add_9: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_8, arg21_1);  mul_8 = arg21_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_16: "f32[16384, 768]" = torch.ops.aten.view.default(add_9, [16384, 768])
        permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg23_1, view_16, permute_10);  arg23_1 = view_16 = permute_10 = None
        view_17: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_6, [128, 128, 768]);  addmm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_18: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_17, [128, -1, 12, 64]);  view_17 = None
        permute_11: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        view_19: "f32[16384, 768]" = torch.ops.aten.view.default(add_9, [16384, 768])
        permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg25_1, view_19, permute_12);  arg25_1 = view_19 = permute_12 = None
        view_20: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_7, [128, 128, 768]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_21: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_20, [128, -1, 12, 64]);  view_20 = None
        permute_13: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:393 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        view_22: "f32[16384, 768]" = torch.ops.aten.view.default(add_9, [16384, 768])
        permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg27_1, view_22, permute_14);  arg27_1 = view_22 = permute_14 = None
        view_23: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_8, [128, 128, 768]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_24: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_23, [128, -1, 12, 64]);  view_23 = None
        permute_15: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:403 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_2, False);  permute_11 = permute_13 = permute_15 = expand_2 = None
        getitem_10: "f32[128, 12, 128, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:389 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_16: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_25: "f32[128, 128, 768]" = torch.ops.aten.view.default(permute_16, [128, -1, 768]);  permute_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:413 in forward, code: attn_output = self.out_lin(attn_output)
        view_26: "f32[16384, 768]" = torch.ops.aten.view.default(view_25, [16384, 768]);  view_25 = None
        permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg29_1, view_26, permute_17);  arg29_1 = view_26 = permute_17 = None
        view_27: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_9, [128, 128, 768]);  addmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:492 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_10: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_27, add_9);  view_27 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 128, 1]" = var_mean_3[0]
        getitem_15: "f32[128, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_11: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_4: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_15);  add_10 = getitem_15 = None
        mul_9: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_10: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg30_1);  mul_9 = arg30_1 = None
        add_12: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_10, arg31_1);  mul_10 = arg31_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:432 in ff_chunk, code: x = self.lin1(input)
        view_28: "f32[16384, 768]" = torch.ops.aten.view.default(add_12, [16384, 768])
        permute_18: "f32[768, 3072]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg33_1, view_28, permute_18);  arg33_1 = view_28 = permute_18 = None
        view_29: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [128, 128, 3072]);  addmm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_11: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:434 in ff_chunk, code: x = self.lin2(x)
        view_30: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_13, [16384, 3072]);  mul_13 = None
        permute_19: "f32[3072, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg35_1, view_30, permute_19);  arg35_1 = view_30 = permute_19 = None
        view_31: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_11, [128, 128, 768]);  addmm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:496 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_14: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_31, add_12);  view_31 = add_12 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_16: "f32[128, 128, 1]" = var_mean_4[0]
        getitem_17: "f32[128, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_15: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_5: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_17);  add_14 = getitem_17 = None
        mul_14: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_15: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg36_1);  mul_14 = arg36_1 = None
        add_16: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_15, arg37_1);  mul_15 = arg37_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_32: "f32[16384, 768]" = torch.ops.aten.view.default(add_16, [16384, 768])
        permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg39_1, view_32, permute_20);  arg39_1 = view_32 = permute_20 = None
        view_33: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_12, [128, 128, 768]);  addmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_34: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_33, [128, -1, 12, 64]);  view_33 = None
        permute_21: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        view_35: "f32[16384, 768]" = torch.ops.aten.view.default(add_16, [16384, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg41_1, view_35, permute_22);  arg41_1 = view_35 = permute_22 = None
        view_36: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_13, [128, 128, 768]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_37: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_36, [128, -1, 12, 64]);  view_36 = None
        permute_23: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:393 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        view_38: "f32[16384, 768]" = torch.ops.aten.view.default(add_16, [16384, 768])
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg43_1, view_38, permute_24);  arg43_1 = view_38 = permute_24 = None
        view_39: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_14, [128, 128, 768]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_40: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_39, [128, -1, 12, 64]);  view_39 = None
        permute_25: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:403 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_3: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_3, False);  permute_21 = permute_23 = permute_25 = expand_3 = None
        getitem_18: "f32[128, 12, 128, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:389 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_26: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        view_41: "f32[128, 128, 768]" = torch.ops.aten.view.default(permute_26, [128, -1, 768]);  permute_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:413 in forward, code: attn_output = self.out_lin(attn_output)
        view_42: "f32[16384, 768]" = torch.ops.aten.view.default(view_41, [16384, 768]);  view_41 = None
        permute_27: "f32[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg45_1, view_42, permute_27);  arg45_1 = view_42 = permute_27 = None
        view_43: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_15, [128, 128, 768]);  addmm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:492 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_17: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_43, add_16);  view_43 = add_16 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22: "f32[128, 128, 1]" = var_mean_5[0]
        getitem_23: "f32[128, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_18: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  add_17 = getitem_23 = None
        mul_16: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_17: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg46_1);  mul_16 = arg46_1 = None
        add_19: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, arg47_1);  mul_17 = arg47_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:432 in ff_chunk, code: x = self.lin1(input)
        view_44: "f32[16384, 768]" = torch.ops.aten.view.default(add_19, [16384, 768])
        permute_28: "f32[768, 3072]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg49_1, view_44, permute_28);  arg49_1 = view_44 = permute_28 = None
        view_45: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [128, 128, 3072]);  addmm_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_18: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:434 in ff_chunk, code: x = self.lin2(x)
        view_46: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_20, [16384, 3072]);  mul_20 = None
        permute_29: "f32[3072, 768]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg51_1, view_46, permute_29);  arg51_1 = view_46 = permute_29 = None
        view_47: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_17, [128, 128, 768]);  addmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:496 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_21: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_47, add_19);  view_47 = add_19 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_24: "f32[128, 128, 1]" = var_mean_6[0]
        getitem_25: "f32[128, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_22: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_7: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_25);  add_21 = getitem_25 = None
        mul_21: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_22: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg52_1);  mul_21 = arg52_1 = None
        add_23: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_22, arg53_1);  mul_22 = arg53_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_48: "f32[16384, 768]" = torch.ops.aten.view.default(add_23, [16384, 768])
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg55_1, view_48, permute_30);  arg55_1 = view_48 = permute_30 = None
        view_49: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_18, [128, 128, 768]);  addmm_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_50: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_49, [128, -1, 12, 64]);  view_49 = None
        permute_31: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        view_51: "f32[16384, 768]" = torch.ops.aten.view.default(add_23, [16384, 768])
        permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg57_1, view_51, permute_32);  arg57_1 = view_51 = permute_32 = None
        view_52: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_19, [128, 128, 768]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_53: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_52, [128, -1, 12, 64]);  view_52 = None
        permute_33: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:393 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        view_54: "f32[16384, 768]" = torch.ops.aten.view.default(add_23, [16384, 768])
        permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg59_1, view_54, permute_34);  arg59_1 = view_54 = permute_34 = None
        view_55: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_20, [128, 128, 768]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_56: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_55, [128, -1, 12, 64]);  view_55 = None
        permute_35: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:403 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_4, False);  permute_31 = permute_33 = permute_35 = expand_4 = None
        getitem_26: "f32[128, 12, 128, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:389 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_36: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        view_57: "f32[128, 128, 768]" = torch.ops.aten.view.default(permute_36, [128, -1, 768]);  permute_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:413 in forward, code: attn_output = self.out_lin(attn_output)
        view_58: "f32[16384, 768]" = torch.ops.aten.view.default(view_57, [16384, 768]);  view_57 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg61_1, view_58, permute_37);  arg61_1 = view_58 = permute_37 = None
        view_59: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_21, [128, 128, 768]);  addmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:492 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_24: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_59, add_23);  view_59 = add_23 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_30: "f32[128, 128, 1]" = var_mean_7[0]
        getitem_31: "f32[128, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_25: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_8: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_31);  add_24 = getitem_31 = None
        mul_23: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_24: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_23, arg62_1);  mul_23 = arg62_1 = None
        add_26: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_24, arg63_1);  mul_24 = arg63_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:432 in ff_chunk, code: x = self.lin1(input)
        view_60: "f32[16384, 768]" = torch.ops.aten.view.default(add_26, [16384, 768])
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg65_1, view_60, permute_38);  arg65_1 = view_60 = permute_38 = None
        view_61: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [128, 128, 3072]);  addmm_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_25: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:434 in ff_chunk, code: x = self.lin2(x)
        view_62: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_27, [16384, 3072]);  mul_27 = None
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg67_1, view_62, permute_39);  arg67_1 = view_62 = permute_39 = None
        view_63: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_23, [128, 128, 768]);  addmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:496 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_28: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_63, add_26);  view_63 = add_26 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_32: "f32[128, 128, 1]" = var_mean_8[0]
        getitem_33: "f32[128, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_29: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_9: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_33);  add_28 = getitem_33 = None
        mul_28: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_29: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg68_1);  mul_28 = arg68_1 = None
        add_30: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_29, arg69_1);  mul_29 = arg69_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_64: "f32[16384, 768]" = torch.ops.aten.view.default(add_30, [16384, 768])
        permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg71_1, view_64, permute_40);  arg71_1 = view_64 = permute_40 = None
        view_65: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_24, [128, 128, 768]);  addmm_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_66: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_65, [128, -1, 12, 64]);  view_65 = None
        permute_41: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        view_67: "f32[16384, 768]" = torch.ops.aten.view.default(add_30, [16384, 768])
        permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg73_1, view_67, permute_42);  arg73_1 = view_67 = permute_42 = None
        view_68: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_25, [128, 128, 768]);  addmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_69: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_68, [128, -1, 12, 64]);  view_68 = None
        permute_43: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:393 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        view_70: "f32[16384, 768]" = torch.ops.aten.view.default(add_30, [16384, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg75_1, view_70, permute_44);  arg75_1 = view_70 = permute_44 = None
        view_71: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_26, [128, 128, 768]);  addmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_72: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_71, [128, -1, 12, 64]);  view_71 = None
        permute_45: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:403 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_5: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(where, [128, 12, 128, 128])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_5, False);  permute_41 = permute_43 = permute_45 = expand_5 = None
        getitem_34: "f32[128, 12, 128, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:389 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_46: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        view_73: "f32[128, 128, 768]" = torch.ops.aten.view.default(permute_46, [128, -1, 768]);  permute_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:413 in forward, code: attn_output = self.out_lin(attn_output)
        view_74: "f32[16384, 768]" = torch.ops.aten.view.default(view_73, [16384, 768]);  view_73 = None
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg77_1, view_74, permute_47);  arg77_1 = view_74 = permute_47 = None
        view_75: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_27, [128, 128, 768]);  addmm_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:492 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_31: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_75, add_30);  view_75 = add_30 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_38: "f32[128, 128, 1]" = var_mean_9[0]
        getitem_39: "f32[128, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_32: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_10: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_39);  add_31 = getitem_39 = None
        mul_30: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_31: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg78_1);  mul_30 = arg78_1 = None
        add_33: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_31, arg79_1);  mul_31 = arg79_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:432 in ff_chunk, code: x = self.lin1(input)
        view_76: "f32[16384, 768]" = torch.ops.aten.view.default(add_33, [16384, 768])
        permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg81_1, view_76, permute_48);  arg81_1 = view_76 = permute_48 = None
        view_77: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [128, 128, 3072]);  addmm_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_32: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:434 in ff_chunk, code: x = self.lin2(x)
        view_78: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_34, [16384, 3072]);  mul_34 = None
        permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg83_1, view_78, permute_49);  arg83_1 = view_78 = permute_49 = None
        view_79: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_29, [128, 128, 768]);  addmm_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:496 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_35: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_79, add_33);  view_79 = add_33 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_40: "f32[128, 128, 1]" = var_mean_10[0]
        getitem_41: "f32[128, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_36: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_11: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_41);  add_35 = getitem_41 = None
        mul_35: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_36: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg84_1);  mul_35 = arg84_1 = None
        add_37: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_36, arg85_1);  mul_36 = arg85_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:391 in forward, code: q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        view_80: "f32[16384, 768]" = torch.ops.aten.view.default(add_37, [16384, 768])
        permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg87_1, view_80, permute_50);  arg87_1 = view_80 = permute_50 = None
        view_81: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_30, [128, 128, 768]);  addmm_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_82: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_81, [128, -1, 12, 64]);  view_81 = None
        permute_51: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:392 in forward, code: k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        view_83: "f32[16384, 768]" = torch.ops.aten.view.default(add_37, [16384, 768])
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg89_1, view_83, permute_52);  arg89_1 = view_83 = permute_52 = None
        view_84: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_31, [128, 128, 768]);  addmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_85: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_84, [128, -1, 12, 64]);  view_84 = None
        permute_53: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:393 in forward, code: v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        view_86: "f32[16384, 768]" = torch.ops.aten.view.default(add_37, [16384, 768])
        permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg91_1, view_86, permute_54);  arg91_1 = view_86 = permute_54 = None
        view_87: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_32, [128, 128, 768]);  addmm_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:385 in shape, code: return x.view(batch_size, -1, self.n_heads, dim_per_head).transpose(1, 2)
        view_88: "f32[128, 128, 12, 64]" = torch.ops.aten.view.default(view_87, [128, -1, 12, 64]);  view_87 = None
        permute_55: "f32[128, 12, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:403 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[128, 12, 128, 128]" = torch.ops.aten.expand.default(where, [128, 12, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_6, False);  permute_51 = permute_53 = permute_55 = expand_6 = None
        getitem_42: "f32[128, 12, 128, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:389 in unshape, code: return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * dim_per_head)
        permute_56: "f32[128, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        view_89: "f32[128, 128, 768]" = torch.ops.aten.view.default(permute_56, [128, -1, 768]);  permute_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:413 in forward, code: attn_output = self.out_lin(attn_output)
        view_90: "f32[16384, 768]" = torch.ops.aten.view.default(view_89, [16384, 768]);  view_89 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg93_1, view_90, permute_57);  arg93_1 = view_90 = permute_57 = None
        view_91: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_33, [128, 128, 768]);  addmm_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:492 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        add_38: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_91, add_37);  view_91 = add_37 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_46: "f32[128, 128, 1]" = var_mean_11[0]
        getitem_47: "f32[128, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_39: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_12: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_47);  add_38 = getitem_47 = None
        mul_37: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_38: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_37, arg94_1);  mul_37 = arg94_1 = None
        add_40: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_38, arg95_1);  mul_38 = arg95_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:432 in ff_chunk, code: x = self.lin1(input)
        view_92: "f32[16384, 768]" = torch.ops.aten.view.default(add_40, [16384, 768])
        permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34: "f32[16384, 3072]" = torch.ops.aten.addmm.default(arg97_1, view_92, permute_58);  arg97_1 = view_92 = permute_58 = None
        view_93: "f32[128, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [128, 128, 3072]);  addmm_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_39: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5: "f32[128, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41: "f32[128, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[128, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:434 in ff_chunk, code: x = self.lin2(x)
        view_94: "f32[16384, 3072]" = torch.ops.aten.view.default(mul_41, [16384, 3072]);  mul_41 = None
        permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg99_1, view_94, permute_59);  arg99_1 = view_94 = permute_59 = None
        view_95: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_35, [128, 128, 768]);  addmm_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:496 in forward, code: ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)
        add_42: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(view_95, add_40);  view_95 = add_40 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_48: "f32[128, 128, 1]" = var_mean_12[0]
        getitem_49: "f32[128, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_43: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_13: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_49);  add_42 = getitem_49 = None
        mul_42: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_43: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg100_1);  mul_42 = arg100_1 = None
        add_44: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, arg101_1);  mul_43 = arg101_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:891 in forward, code: prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        view_96: "f32[16384, 768]" = torch.ops.aten.view.default(add_44, [16384, 768]);  add_44 = None
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36: "f32[16384, 768]" = torch.ops.aten.addmm.default(arg103_1, view_96, permute_60);  arg103_1 = view_96 = permute_60 = None
        view_97: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_36, [128, 128, 768]);  addmm_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_44: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(view_97, 0.5)
        mul_45: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(view_97, 0.7071067811865476);  view_97 = None
        erf_6: "f32[128, 128, 768]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_45: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_44, add_45);  mul_44 = add_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:893 in forward, code: prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        var_mean_13 = torch.ops.aten.var_mean.correction(mul_46, [2], correction = 0, keepdim = True)
        getitem_50: "f32[128, 128, 1]" = var_mean_13[0]
        getitem_51: "f32[128, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_46: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_13: "f32[128, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_14: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(mul_46, getitem_51);  mul_46 = getitem_51 = None
        mul_47: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_48: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_47, arg104_1);  mul_47 = arg104_1 = None
        add_47: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(mul_48, arg105_1);  mul_48 = arg105_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:894 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_98: "f32[16384, 768]" = torch.ops.aten.view.default(add_47, [16384, 768]);  add_47 = None
        permute_61: "f32[768, 30522]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        
        # No stacktrace found for following nodes
        full_default_2: "f32[768, 2]" = torch.ops.aten.full.default([768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[768, 30524]" = torch.ops.aten.cat.default([permute_61, full_default_2], 1);  permute_61 = full_default_2 = None
        full_default_3: "f32[2]" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default_1: "f32[30524]" = torch.ops.aten.cat.default([arg106_1, full_default_3]);  arg106_1 = full_default_3 = None
        addmm_default: "f32[16384, 30524]" = torch.ops.aten.addmm.default(cat_default_1, view_98, cat_default);  cat_default_1 = view_98 = cat_default = None
        slice_tensor: "f32[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -2);  addmm_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:894 in forward, code: prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        view_99: "f32[128, 128, 30522]" = torch.ops.aten.view.default(slice_tensor, [128, 128, 30522]);  slice_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/distilbert/modeling_distilbert.py:898 in forward, code: mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        view_100: "f32[16384, 30522]" = torch.ops.aten.view.default(view_99, [-1, 30522])
        view_101: "i64[16384]" = torch.ops.aten.view.default(arg107_1, [-1]);  arg107_1 = None
        amax: "f32[16384, 1]" = torch.ops.aten.amax.default(view_100, [1], True)
        sub_15: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(view_100, amax);  view_100 = amax = None
        exp: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_15)
        sum_1: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_16: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_15, log);  sub_15 = log = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_101, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[16384]" = torch.ops.aten.where.self(ne, view_101, full_default);  ne = full_default = None
        unsqueeze_2: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_16, 1, unsqueeze_2);  sub_16 = unsqueeze_2 = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1: "b8[16384]" = torch.ops.aten.ne.Scalar(view_101, -100)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[16384]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2: "b8[16384]" = torch.ops.aten.ne.Scalar(view_101, -100);  view_101 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        return (div, view_99)
        