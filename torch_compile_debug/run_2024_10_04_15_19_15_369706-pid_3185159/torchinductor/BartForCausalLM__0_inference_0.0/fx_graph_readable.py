class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 1024]", arg1_1: "f32[50265, 1024]", arg2_1: "f32[1026, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024, 1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg8_1: "f32[1024]", arg9_1: "f32[1024, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[1024, 1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024]", arg14_1: "f32[1024]", arg15_1: "f32[4096, 1024]", arg16_1: "f32[4096]", arg17_1: "f32[1024, 4096]", arg18_1: "f32[1024]", arg19_1: "f32[1024]", arg20_1: "f32[1024]", arg21_1: "f32[1024, 1024]", arg22_1: "f32[1024]", arg23_1: "f32[1024, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024, 1024]", arg26_1: "f32[1024]", arg27_1: "f32[1024, 1024]", arg28_1: "f32[1024]", arg29_1: "f32[1024]", arg30_1: "f32[1024]", arg31_1: "f32[4096, 1024]", arg32_1: "f32[4096]", arg33_1: "f32[1024, 4096]", arg34_1: "f32[1024]", arg35_1: "f32[1024]", arg36_1: "f32[1024]", arg37_1: "f32[1024, 1024]", arg38_1: "f32[1024]", arg39_1: "f32[1024, 1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024, 1024]", arg42_1: "f32[1024]", arg43_1: "f32[1024, 1024]", arg44_1: "f32[1024]", arg45_1: "f32[1024]", arg46_1: "f32[1024]", arg47_1: "f32[4096, 1024]", arg48_1: "f32[4096]", arg49_1: "f32[1024, 4096]", arg50_1: "f32[1024]", arg51_1: "f32[1024]", arg52_1: "f32[1024]", arg53_1: "f32[1024, 1024]", arg54_1: "f32[1024]", arg55_1: "f32[1024, 1024]", arg56_1: "f32[1024]", arg57_1: "f32[1024, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024, 1024]", arg60_1: "f32[1024]", arg61_1: "f32[1024]", arg62_1: "f32[1024]", arg63_1: "f32[4096, 1024]", arg64_1: "f32[4096]", arg65_1: "f32[1024, 4096]", arg66_1: "f32[1024]", arg67_1: "f32[1024]", arg68_1: "f32[1024]", arg69_1: "f32[1024, 1024]", arg70_1: "f32[1024]", arg71_1: "f32[1024, 1024]", arg72_1: "f32[1024]", arg73_1: "f32[1024, 1024]", arg74_1: "f32[1024]", arg75_1: "f32[1024, 1024]", arg76_1: "f32[1024]", arg77_1: "f32[1024]", arg78_1: "f32[1024]", arg79_1: "f32[4096, 1024]", arg80_1: "f32[4096]", arg81_1: "f32[1024, 4096]", arg82_1: "f32[1024]", arg83_1: "f32[1024]", arg84_1: "f32[1024]", arg85_1: "f32[1024, 1024]", arg86_1: "f32[1024]", arg87_1: "f32[1024, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 1024]", arg90_1: "f32[1024]", arg91_1: "f32[1024, 1024]", arg92_1: "f32[1024]", arg93_1: "f32[1024]", arg94_1: "f32[1024]", arg95_1: "f32[4096, 1024]", arg96_1: "f32[4096]", arg97_1: "f32[1024, 4096]", arg98_1: "f32[1024]", arg99_1: "f32[1024]", arg100_1: "f32[1024]", arg101_1: "f32[1024, 1024]", arg102_1: "f32[1024]", arg103_1: "f32[1024, 1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024, 1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024, 1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024]", arg110_1: "f32[1024]", arg111_1: "f32[4096, 1024]", arg112_1: "f32[4096]", arg113_1: "f32[1024, 4096]", arg114_1: "f32[1024]", arg115_1: "f32[1024]", arg116_1: "f32[1024]", arg117_1: "f32[1024, 1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024, 1024]", arg122_1: "f32[1024]", arg123_1: "f32[1024, 1024]", arg124_1: "f32[1024]", arg125_1: "f32[1024]", arg126_1: "f32[1024]", arg127_1: "f32[4096, 1024]", arg128_1: "f32[4096]", arg129_1: "f32[1024, 4096]", arg130_1: "f32[1024]", arg131_1: "f32[1024]", arg132_1: "f32[1024]", arg133_1: "f32[1024, 1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024, 1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024, 1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024]", arg142_1: "f32[1024]", arg143_1: "f32[4096, 1024]", arg144_1: "f32[4096]", arg145_1: "f32[1024, 4096]", arg146_1: "f32[1024]", arg147_1: "f32[1024]", arg148_1: "f32[1024]", arg149_1: "f32[1024, 1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024, 1024]", arg152_1: "f32[1024]", arg153_1: "f32[1024, 1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024, 1024]", arg156_1: "f32[1024]", arg157_1: "f32[1024]", arg158_1: "f32[1024]", arg159_1: "f32[4096, 1024]", arg160_1: "f32[4096]", arg161_1: "f32[1024, 4096]", arg162_1: "f32[1024]", arg163_1: "f32[1024]", arg164_1: "f32[1024]", arg165_1: "f32[1024, 1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024, 1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024, 1024]", arg172_1: "f32[1024]", arg173_1: "f32[1024]", arg174_1: "f32[1024]", arg175_1: "f32[4096, 1024]", arg176_1: "f32[4096]", arg177_1: "f32[1024, 4096]", arg178_1: "f32[1024]", arg179_1: "f32[1024]", arg180_1: "f32[1024]", arg181_1: "f32[1024, 1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024, 1024]", arg186_1: "f32[1024]", arg187_1: "f32[1024, 1024]", arg188_1: "f32[1024]", arg189_1: "f32[1024]", arg190_1: "f32[1024]", arg191_1: "f32[4096, 1024]", arg192_1: "f32[4096]", arg193_1: "f32[1024, 4096]", arg194_1: "f32[1024]", arg195_1: "f32[1024]", arg196_1: "f32[1024]", arg197_1: "i64[4, 1024]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:129 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[4, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 1);  arg0_1 = None
        mul: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:158 in _make_causal_mask, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        full_default: "f32[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:159 in _make_causal_mask, code: mask_cond = torch.arange(mask.size(-1), device=device)
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:160 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        add: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 1)
        view_1: "i64[1024, 1]" = torch.ops.aten.view.default(add, [1024, 1]);  add = None
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1024, 1024]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:112 in forward, code: positions = torch.arange(
        iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:114 in forward, code: ).expand(bsz, -1)
        expand_1: "i64[4, 1024]" = torch.ops.aten.expand.default(iota_1, [4, -1]);  iota_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:116 in forward, code: return super().forward(positions + self.offset)
        add_1: "i64[4, 1024]" = torch.ops.aten.add.Tensor(expand_1, 2);  expand_1 = None
        embedding_1: "f32[4, 1024, 1024]" = torch.ops.aten.embedding.default(arg2_1, add_1);  arg2_1 = add_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1328 in forward, code: hidden_states = inputs_embeds + positions
        add_2: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:1329 in forward, code: hidden_states = self.layernorm_embedding(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 1024, 1]" = var_mean[0]
        getitem_1: "f32[4, 1024, 1]" = var_mean[1];  var_mean = None
        add_3: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
        mul_1: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_4: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_2: "f32[4096, 1024]" = torch.ops.aten.view.default(add_4, [4096, 1024])
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg6_1, view_2, permute);  arg6_1 = view_2 = permute = None
        view_3: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm, [4, 1024, 1024]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_4: "f32[4096, 1024]" = torch.ops.aten.view.default(add_4, [4096, 1024])
        permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg8_1, view_4, permute_1);  arg8_1 = view_4 = permute_1 = None
        view_5: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_1, [4, 1024, 1024]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_6: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_5, [4, -1, 16, 64]);  view_5 = None
        permute_2: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_7: "f32[4096, 1024]" = torch.ops.aten.view.default(add_4, [4096, 1024])
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_7, permute_3);  arg10_1 = view_7 = permute_3 = None
        view_8: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_2, [4, 1024, 1024]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_9: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_8, [4, -1, 16, 64]);  view_8 = None
        permute_4: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_2: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_10: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_3, [4, 1024, 16, 64]);  view_3 = None
        permute_5: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_2: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_3: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_3: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_3, [4, 1, 1024, 1024]);  unsqueeze_3 = None
        expand_4: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_3, [4, 16, 1024, 1024]);  expand_3 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_3, clone_1, clone_2, expand_4, False);  clone_3 = expand_4 = None
        getitem_2: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_6: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_11: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_6, [4, 1024, 1024]);  permute_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_12: "f32[4096, 1024]" = torch.ops.aten.view.default(view_11, [4096, 1024]);  view_11 = None
        permute_7: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg12_1, view_12, permute_7);  arg12_1 = view_12 = permute_7 = None
        view_13: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_3, [4, 1024, 1024]);  addmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_5: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_4, view_13);  add_4 = view_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_6: "f32[4, 1024, 1]" = var_mean_1[0]
        getitem_7: "f32[4, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_1: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_7);  add_5 = getitem_7 = None
        mul_3: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_4: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_3, arg13_1);  mul_3 = arg13_1 = None
        add_7: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_14: "f32[4096, 1024]" = torch.ops.aten.view.default(add_7, [4096, 1024])
        permute_8: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg16_1, view_14, permute_8);  arg16_1 = view_14 = permute_8 = None
        view_15: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_4, [4, 1024, 4096]);  addmm_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_5: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
        mul_6: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
        erf: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_8: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_5, add_8);  mul_5 = add_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_16: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_7, [4096, 4096]);  mul_7 = None
        permute_9: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg18_1, view_16, permute_9);  arg18_1 = view_16 = permute_9 = None
        view_17: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_5, [4, 1024, 1024]);  addmm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_9: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_7, view_17);  add_7 = view_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_8: "f32[4, 1024, 1]" = var_mean_2[0]
        getitem_9: "f32[4, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_9, getitem_9);  add_9 = getitem_9 = None
        mul_8: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_8, arg19_1);  mul_8 = arg19_1 = None
        add_11: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_18: "f32[4096, 1024]" = torch.ops.aten.view.default(add_11, [4096, 1024])
        permute_10: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg22_1, view_18, permute_10);  arg22_1 = view_18 = permute_10 = None
        view_19: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_6, [4, 1024, 1024]);  addmm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_20: "f32[4096, 1024]" = torch.ops.aten.view.default(add_11, [4096, 1024])
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg24_1, view_20, permute_11);  arg24_1 = view_20 = permute_11 = None
        view_21: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_7, [4, 1024, 1024]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_22: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_21, [4, -1, 16, 64]);  view_21 = None
        permute_12: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        clone_7: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_23: "f32[4096, 1024]" = torch.ops.aten.view.default(add_11, [4096, 1024])
        permute_13: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg26_1, view_23, permute_13);  arg26_1 = view_23 = permute_13 = None
        view_24: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_8, [4, 1024, 1024]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_25: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_24, [4, -1, 16, 64]);  view_24 = None
        permute_14: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        clone_8: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_26: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_19, [4, 1024, 16, 64]);  view_19 = None
        permute_15: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        clone_9: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_4: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_5: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        expand_6: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_5, [4, 1, 1024, 1024]);  unsqueeze_5 = None
        expand_7: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_6, [4, 16, 1024, 1024]);  expand_6 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_9, clone_7, clone_8, expand_7, False);  clone_9 = expand_7 = None
        getitem_10: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_16: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_27: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_16, [4, 1024, 1024]);  permute_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_28: "f32[4096, 1024]" = torch.ops.aten.view.default(view_27, [4096, 1024]);  view_27 = None
        permute_17: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg28_1, view_28, permute_17);  arg28_1 = view_28 = permute_17 = None
        view_29: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_9, [4, 1024, 1024]);  addmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_12: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_11, view_29);  add_11 = view_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_14: "f32[4, 1024, 1]" = var_mean_3[0]
        getitem_15: "f32[4, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_3: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_12, getitem_15);  add_12 = getitem_15 = None
        mul_10: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_10, arg29_1);  mul_10 = arg29_1 = None
        add_14: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_11, arg30_1);  mul_11 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_30: "f32[4096, 1024]" = torch.ops.aten.view.default(add_14, [4096, 1024])
        permute_18: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg32_1, view_30, permute_18);  arg32_1 = view_30 = permute_18 = None
        view_31: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_10, [4, 1024, 4096]);  addmm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_12: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
        mul_13: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476);  view_31 = None
        erf_1: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_15: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_32: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_14, [4096, 4096]);  mul_14 = None
        permute_19: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg34_1, view_32, permute_19);  arg34_1 = view_32 = permute_19 = None
        view_33: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_11, [4, 1024, 1024]);  addmm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_16: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_14, view_33);  add_14 = view_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_16: "f32[4, 1024, 1]" = var_mean_4[0]
        getitem_17: "f32[4, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        add_17: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_4: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_16, getitem_17);  add_16 = getitem_17 = None
        mul_15: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_16: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_15, arg35_1);  mul_15 = arg35_1 = None
        add_18: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_16, arg36_1);  mul_16 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_34: "f32[4096, 1024]" = torch.ops.aten.view.default(add_18, [4096, 1024])
        permute_20: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg38_1, view_34, permute_20);  arg38_1 = view_34 = permute_20 = None
        view_35: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_12, [4, 1024, 1024]);  addmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_36: "f32[4096, 1024]" = torch.ops.aten.view.default(add_18, [4096, 1024])
        permute_21: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg40_1, view_36, permute_21);  arg40_1 = view_36 = permute_21 = None
        view_37: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_13, [4, 1024, 1024]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_38: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_37, [4, -1, 16, 64]);  view_37 = None
        permute_22: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_13: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_39: "f32[4096, 1024]" = torch.ops.aten.view.default(add_18, [4096, 1024])
        permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg42_1, view_39, permute_23);  arg42_1 = view_39 = permute_23 = None
        view_40: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_14, [4, 1024, 1024]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_41: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_40, [4, -1, 16, 64]);  view_40 = None
        permute_24: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_14: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_42: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_35, [4, 1024, 16, 64]);  view_35 = None
        permute_25: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        clone_15: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_6: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_7: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        expand_9: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_7, [4, 1, 1024, 1024]);  unsqueeze_7 = None
        expand_10: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_9, [4, 16, 1024, 1024]);  expand_9 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_15, clone_13, clone_14, expand_10, False);  clone_15 = expand_10 = None
        getitem_18: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_26: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_43: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_26, [4, 1024, 1024]);  permute_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_44: "f32[4096, 1024]" = torch.ops.aten.view.default(view_43, [4096, 1024]);  view_43 = None
        permute_27: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg44_1, view_44, permute_27);  arg44_1 = view_44 = permute_27 = None
        view_45: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_15, [4, 1024, 1024]);  addmm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_19: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_18, view_45);  add_18 = view_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_22: "f32[4, 1024, 1]" = var_mean_5[0]
        getitem_23: "f32[4, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        add_20: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_5: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_5: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_19, getitem_23);  add_19 = getitem_23 = None
        mul_17: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_18: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg45_1);  mul_17 = arg45_1 = None
        add_21: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg46_1);  mul_18 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_46: "f32[4096, 1024]" = torch.ops.aten.view.default(add_21, [4096, 1024])
        permute_28: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg48_1, view_46, permute_28);  arg48_1 = view_46 = permute_28 = None
        view_47: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_16, [4, 1024, 4096]);  addmm_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_19: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
        mul_20: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_47, 0.7071067811865476);  view_47 = None
        erf_2: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_22: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_19, add_22);  mul_19 = add_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_48: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_21, [4096, 4096]);  mul_21 = None
        permute_29: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg50_1, view_48, permute_29);  arg50_1 = view_48 = permute_29 = None
        view_49: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_17, [4, 1024, 1024]);  addmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_23: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_21, view_49);  add_21 = view_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_24: "f32[4, 1024, 1]" = var_mean_6[0]
        getitem_25: "f32[4, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        add_24: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_6: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_23, getitem_25);  add_23 = getitem_25 = None
        mul_22: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_23: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_22, arg51_1);  mul_22 = arg51_1 = None
        add_25: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_23, arg52_1);  mul_23 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_50: "f32[4096, 1024]" = torch.ops.aten.view.default(add_25, [4096, 1024])
        permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg54_1, view_50, permute_30);  arg54_1 = view_50 = permute_30 = None
        view_51: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_18, [4, 1024, 1024]);  addmm_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_52: "f32[4096, 1024]" = torch.ops.aten.view.default(add_25, [4096, 1024])
        permute_31: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg56_1, view_52, permute_31);  arg56_1 = view_52 = permute_31 = None
        view_53: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_19, [4, 1024, 1024]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_54: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_53, [4, -1, 16, 64]);  view_53 = None
        permute_32: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_19: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_55: "f32[4096, 1024]" = torch.ops.aten.view.default(add_25, [4096, 1024])
        permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg58_1, view_55, permute_33);  arg58_1 = view_55 = permute_33 = None
        view_56: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_20, [4, 1024, 1024]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_57: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_56, [4, -1, 16, 64]);  view_56 = None
        permute_34: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        clone_20: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_58: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_51, [4, 1024, 16, 64]);  view_51 = None
        permute_35: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_21: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_8: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_9: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
        expand_12: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_9, [4, 1, 1024, 1024]);  unsqueeze_9 = None
        expand_13: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_12, [4, 16, 1024, 1024]);  expand_12 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_21, clone_19, clone_20, expand_13, False);  clone_21 = expand_13 = None
        getitem_26: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_36: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_59: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_36, [4, 1024, 1024]);  permute_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_60: "f32[4096, 1024]" = torch.ops.aten.view.default(view_59, [4096, 1024]);  view_59 = None
        permute_37: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg60_1, view_60, permute_37);  arg60_1 = view_60 = permute_37 = None
        view_61: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_21, [4, 1024, 1024]);  addmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_26: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_25, view_61);  add_25 = view_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_30: "f32[4, 1024, 1]" = var_mean_7[0]
        getitem_31: "f32[4, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        add_27: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_7: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_7: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_26, getitem_31);  add_26 = getitem_31 = None
        mul_24: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_25: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_24, arg61_1);  mul_24 = arg61_1 = None
        add_28: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_25, arg62_1);  mul_25 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_62: "f32[4096, 1024]" = torch.ops.aten.view.default(add_28, [4096, 1024])
        permute_38: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg64_1, view_62, permute_38);  arg64_1 = view_62 = permute_38 = None
        view_63: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_22, [4, 1024, 4096]);  addmm_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_26: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_27: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
        erf_3: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_29: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_28: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_26, add_29);  mul_26 = add_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_64: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_28, [4096, 4096]);  mul_28 = None
        permute_39: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg66_1, view_64, permute_39);  arg66_1 = view_64 = permute_39 = None
        view_65: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_23, [4, 1024, 1024]);  addmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_30: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_28, view_65);  add_28 = view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_32: "f32[4, 1024, 1]" = var_mean_8[0]
        getitem_33: "f32[4, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        add_31: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_8: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_30, getitem_33);  add_30 = getitem_33 = None
        mul_29: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_30: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_29, arg67_1);  mul_29 = arg67_1 = None
        add_32: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_30, arg68_1);  mul_30 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_66: "f32[4096, 1024]" = torch.ops.aten.view.default(add_32, [4096, 1024])
        permute_40: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg70_1, view_66, permute_40);  arg70_1 = view_66 = permute_40 = None
        view_67: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_24, [4, 1024, 1024]);  addmm_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_68: "f32[4096, 1024]" = torch.ops.aten.view.default(add_32, [4096, 1024])
        permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg72_1, view_68, permute_41);  arg72_1 = view_68 = permute_41 = None
        view_69: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_25, [4, 1024, 1024]);  addmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_70: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_69, [4, -1, 16, 64]);  view_69 = None
        permute_42: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_25: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_71: "f32[4096, 1024]" = torch.ops.aten.view.default(add_32, [4096, 1024])
        permute_43: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_71, permute_43);  arg74_1 = view_71 = permute_43 = None
        view_72: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_26, [4, 1024, 1024]);  addmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_73: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_72, [4, -1, 16, 64]);  view_72 = None
        permute_44: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        clone_26: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_74: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_67, [4, 1024, 16, 64]);  view_67 = None
        permute_45: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        clone_27: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_10: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_11: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
        expand_15: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_11, [4, 1, 1024, 1024]);  unsqueeze_11 = None
        expand_16: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_15, [4, 16, 1024, 1024]);  expand_15 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_27, clone_25, clone_26, expand_16, False);  clone_27 = expand_16 = None
        getitem_34: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_46: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_75: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_46, [4, 1024, 1024]);  permute_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_76: "f32[4096, 1024]" = torch.ops.aten.view.default(view_75, [4096, 1024]);  view_75 = None
        permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg76_1, view_76, permute_47);  arg76_1 = view_76 = permute_47 = None
        view_77: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_27, [4, 1024, 1024]);  addmm_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_33: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_32, view_77);  add_32 = view_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_38: "f32[4, 1024, 1]" = var_mean_9[0]
        getitem_39: "f32[4, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        add_34: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_9: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_9: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_33, getitem_39);  add_33 = getitem_39 = None
        mul_31: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_32: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_31, arg77_1);  mul_31 = arg77_1 = None
        add_35: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_32, arg78_1);  mul_32 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_78: "f32[4096, 1024]" = torch.ops.aten.view.default(add_35, [4096, 1024])
        permute_48: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg80_1, view_78, permute_48);  arg80_1 = view_78 = permute_48 = None
        view_79: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_28, [4, 1024, 4096]);  addmm_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_33: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_79, 0.5)
        mul_34: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_79, 0.7071067811865476);  view_79 = None
        erf_4: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_36: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_35: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_33, add_36);  mul_33 = add_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_80: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_35, [4096, 4096]);  mul_35 = None
        permute_49: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg82_1, view_80, permute_49);  arg82_1 = view_80 = permute_49 = None
        view_81: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_29, [4, 1024, 1024]);  addmm_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_37: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_35, view_81);  add_35 = view_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_40: "f32[4, 1024, 1]" = var_mean_10[0]
        getitem_41: "f32[4, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        add_38: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_10: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_37, getitem_41);  add_37 = getitem_41 = None
        mul_36: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_37: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg83_1);  mul_36 = arg83_1 = None
        add_39: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg84_1);  mul_37 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_82: "f32[4096, 1024]" = torch.ops.aten.view.default(add_39, [4096, 1024])
        permute_50: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg86_1, view_82, permute_50);  arg86_1 = view_82 = permute_50 = None
        view_83: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_30, [4, 1024, 1024]);  addmm_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_84: "f32[4096, 1024]" = torch.ops.aten.view.default(add_39, [4096, 1024])
        permute_51: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg88_1, view_84, permute_51);  arg88_1 = view_84 = permute_51 = None
        view_85: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_31, [4, 1024, 1024]);  addmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_86: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_85, [4, -1, 16, 64]);  view_85 = None
        permute_52: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        clone_31: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_87: "f32[4096, 1024]" = torch.ops.aten.view.default(add_39, [4096, 1024])
        permute_53: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg90_1, view_87, permute_53);  arg90_1 = view_87 = permute_53 = None
        view_88: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_32, [4, 1024, 1024]);  addmm_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_89: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_88, [4, -1, 16, 64]);  view_88 = None
        permute_54: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_32: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_90: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_83, [4, 1024, 16, 64]);  view_83 = None
        permute_55: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_33: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_12: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_13: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        expand_18: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_13, [4, 1, 1024, 1024]);  unsqueeze_13 = None
        expand_19: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_18, [4, 16, 1024, 1024]);  expand_18 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_33, clone_31, clone_32, expand_19, False);  clone_33 = expand_19 = None
        getitem_42: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_56: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_91: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_56, [4, 1024, 1024]);  permute_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_92: "f32[4096, 1024]" = torch.ops.aten.view.default(view_91, [4096, 1024]);  view_91 = None
        permute_57: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg92_1, view_92, permute_57);  arg92_1 = view_92 = permute_57 = None
        view_93: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_33, [4, 1024, 1024]);  addmm_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_40: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_39, view_93);  add_39 = view_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_46: "f32[4, 1024, 1]" = var_mean_11[0]
        getitem_47: "f32[4, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        add_41: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_11: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_11: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_40, getitem_47);  add_40 = getitem_47 = None
        mul_38: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_39: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_38, arg93_1);  mul_38 = arg93_1 = None
        add_42: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_39, arg94_1);  mul_39 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_94: "f32[4096, 1024]" = torch.ops.aten.view.default(add_42, [4096, 1024])
        permute_58: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg96_1, view_94, permute_58);  arg96_1 = view_94 = permute_58 = None
        view_95: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_34, [4, 1024, 4096]);  addmm_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_40: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_95, 0.5)
        mul_41: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_95, 0.7071067811865476);  view_95 = None
        erf_5: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_43: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_42: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_40, add_43);  mul_40 = add_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_96: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_42, [4096, 4096]);  mul_42 = None
        permute_59: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg98_1, view_96, permute_59);  arg98_1 = view_96 = permute_59 = None
        view_97: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_35, [4, 1024, 1024]);  addmm_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_44: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_42, view_97);  add_42 = view_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_48: "f32[4, 1024, 1]" = var_mean_12[0]
        getitem_49: "f32[4, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        add_45: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_12: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_44, getitem_49);  add_44 = getitem_49 = None
        mul_43: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_44: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_43, arg99_1);  mul_43 = arg99_1 = None
        add_46: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_44, arg100_1);  mul_44 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_98: "f32[4096, 1024]" = torch.ops.aten.view.default(add_46, [4096, 1024])
        permute_60: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg102_1, view_98, permute_60);  arg102_1 = view_98 = permute_60 = None
        view_99: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_36, [4, 1024, 1024]);  addmm_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_100: "f32[4096, 1024]" = torch.ops.aten.view.default(add_46, [4096, 1024])
        permute_61: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_100, permute_61);  arg104_1 = view_100 = permute_61 = None
        view_101: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_37, [4, 1024, 1024]);  addmm_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_102: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_101, [4, -1, 16, 64]);  view_101 = None
        permute_62: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_37: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_103: "f32[4096, 1024]" = torch.ops.aten.view.default(add_46, [4096, 1024])
        permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg106_1, view_103, permute_63);  arg106_1 = view_103 = permute_63 = None
        view_104: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_38, [4, 1024, 1024]);  addmm_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_105: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_104, [4, -1, 16, 64]);  view_104 = None
        permute_64: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
        clone_38: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_106: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_99, [4, 1024, 16, 64]);  view_99 = None
        permute_65: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_39: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_14: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_15: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
        expand_21: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_15, [4, 1, 1024, 1024]);  unsqueeze_15 = None
        expand_22: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_21, [4, 16, 1024, 1024]);  expand_21 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_39, clone_37, clone_38, expand_22, False);  clone_39 = expand_22 = None
        getitem_50: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_66: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_107: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_66, [4, 1024, 1024]);  permute_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_108: "f32[4096, 1024]" = torch.ops.aten.view.default(view_107, [4096, 1024]);  view_107 = None
        permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg108_1, view_108, permute_67);  arg108_1 = view_108 = permute_67 = None
        view_109: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_39, [4, 1024, 1024]);  addmm_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_47: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_46, view_109);  add_46 = view_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
        getitem_54: "f32[4, 1024, 1]" = var_mean_13[0]
        getitem_55: "f32[4, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        add_48: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_13: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_13: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_47, getitem_55);  add_47 = getitem_55 = None
        mul_45: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_46: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_45, arg109_1);  mul_45 = arg109_1 = None
        add_49: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_46, arg110_1);  mul_46 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_110: "f32[4096, 1024]" = torch.ops.aten.view.default(add_49, [4096, 1024])
        permute_68: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg112_1, view_110, permute_68);  arg112_1 = view_110 = permute_68 = None
        view_111: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_40, [4, 1024, 4096]);  addmm_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_47: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
        mul_48: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476);  view_111 = None
        erf_6: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_50: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_49: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_47, add_50);  mul_47 = add_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_112: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_49, [4096, 4096]);  mul_49 = None
        permute_69: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg114_1, view_112, permute_69);  arg114_1 = view_112 = permute_69 = None
        view_113: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_41, [4, 1024, 1024]);  addmm_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_51: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_49, view_113);  add_49 = view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_56: "f32[4, 1024, 1]" = var_mean_14[0]
        getitem_57: "f32[4, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        add_52: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_14: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_51, getitem_57);  add_51 = getitem_57 = None
        mul_50: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_51: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_50, arg115_1);  mul_50 = arg115_1 = None
        add_53: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_51, arg116_1);  mul_51 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_114: "f32[4096, 1024]" = torch.ops.aten.view.default(add_53, [4096, 1024])
        permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg118_1, view_114, permute_70);  arg118_1 = view_114 = permute_70 = None
        view_115: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_42, [4, 1024, 1024]);  addmm_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_116: "f32[4096, 1024]" = torch.ops.aten.view.default(add_53, [4096, 1024])
        permute_71: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg120_1, view_116, permute_71);  arg120_1 = view_116 = permute_71 = None
        view_117: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_43, [4, 1024, 1024]);  addmm_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_118: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_117, [4, -1, 16, 64]);  view_117 = None
        permute_72: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_43: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_119: "f32[4096, 1024]" = torch.ops.aten.view.default(add_53, [4096, 1024])
        permute_73: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg122_1, view_119, permute_73);  arg122_1 = view_119 = permute_73 = None
        view_120: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_44, [4, 1024, 1024]);  addmm_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_121: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_120, [4, -1, 16, 64]);  view_120 = None
        permute_74: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_44: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_122: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_115, [4, 1024, 16, 64]);  view_115 = None
        permute_75: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        clone_45: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_16: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_17: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        expand_24: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_17, [4, 1, 1024, 1024]);  unsqueeze_17 = None
        expand_25: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_24, [4, 16, 1024, 1024]);  expand_24 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_45, clone_43, clone_44, expand_25, False);  clone_45 = expand_25 = None
        getitem_58: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_76: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_123: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_76, [4, 1024, 1024]);  permute_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_124: "f32[4096, 1024]" = torch.ops.aten.view.default(view_123, [4096, 1024]);  view_123 = None
        permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg124_1, view_124, permute_77);  arg124_1 = view_124 = permute_77 = None
        view_125: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_45, [4, 1024, 1024]);  addmm_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_54: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_53, view_125);  add_53 = view_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
        getitem_62: "f32[4, 1024, 1]" = var_mean_15[0]
        getitem_63: "f32[4, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        add_55: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_15: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_15: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_54, getitem_63);  add_54 = getitem_63 = None
        mul_52: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_53: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg125_1);  mul_52 = arg125_1 = None
        add_56: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg126_1);  mul_53 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_126: "f32[4096, 1024]" = torch.ops.aten.view.default(add_56, [4096, 1024])
        permute_78: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg128_1, view_126, permute_78);  arg128_1 = view_126 = permute_78 = None
        view_127: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_46, [4, 1024, 4096]);  addmm_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_54: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_127, 0.5)
        mul_55: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476);  view_127 = None
        erf_7: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_57: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_56: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_57);  mul_54 = add_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_128: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_56, [4096, 4096]);  mul_56 = None
        permute_79: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg130_1, view_128, permute_79);  arg130_1 = view_128 = permute_79 = None
        view_129: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_47, [4, 1024, 1024]);  addmm_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_58: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_56, view_129);  add_56 = view_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
        getitem_64: "f32[4, 1024, 1]" = var_mean_16[0]
        getitem_65: "f32[4, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        add_59: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_16: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_58, getitem_65);  add_58 = getitem_65 = None
        mul_57: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_58: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg131_1);  mul_57 = arg131_1 = None
        add_60: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg132_1);  mul_58 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_130: "f32[4096, 1024]" = torch.ops.aten.view.default(add_60, [4096, 1024])
        permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg134_1, view_130, permute_80);  arg134_1 = view_130 = permute_80 = None
        view_131: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_48, [4, 1024, 1024]);  addmm_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_132: "f32[4096, 1024]" = torch.ops.aten.view.default(add_60, [4096, 1024])
        permute_81: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg136_1, view_132, permute_81);  arg136_1 = view_132 = permute_81 = None
        view_133: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_49, [4, 1024, 1024]);  addmm_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_134: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_133, [4, -1, 16, 64]);  view_133 = None
        permute_82: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        clone_49: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_135: "f32[4096, 1024]" = torch.ops.aten.view.default(add_60, [4096, 1024])
        permute_83: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_135, permute_83);  arg138_1 = view_135 = permute_83 = None
        view_136: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_50, [4, 1024, 1024]);  addmm_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_137: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_136, [4, -1, 16, 64]);  view_136 = None
        permute_84: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_50: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_138: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_131, [4, 1024, 16, 64]);  view_131 = None
        permute_85: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_51: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_18: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_19: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
        expand_27: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_19, [4, 1, 1024, 1024]);  unsqueeze_19 = None
        expand_28: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_27, [4, 16, 1024, 1024]);  expand_27 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_51, clone_49, clone_50, expand_28, False);  clone_51 = expand_28 = None
        getitem_66: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_86: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_139: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_86, [4, 1024, 1024]);  permute_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_140: "f32[4096, 1024]" = torch.ops.aten.view.default(view_139, [4096, 1024]);  view_139 = None
        permute_87: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg140_1, view_140, permute_87);  arg140_1 = view_140 = permute_87 = None
        view_141: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_51, [4, 1024, 1024]);  addmm_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_61: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_60, view_141);  add_60 = view_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_70: "f32[4, 1024, 1]" = var_mean_17[0]
        getitem_71: "f32[4, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        add_62: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_17: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_17: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_71);  add_61 = getitem_71 = None
        mul_59: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_60: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_59, arg141_1);  mul_59 = arg141_1 = None
        add_63: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_60, arg142_1);  mul_60 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_142: "f32[4096, 1024]" = torch.ops.aten.view.default(add_63, [4096, 1024])
        permute_88: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg144_1, view_142, permute_88);  arg144_1 = view_142 = permute_88 = None
        view_143: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_52, [4, 1024, 4096]);  addmm_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_61: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
        mul_62: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_143, 0.7071067811865476);  view_143 = None
        erf_8: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_64: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_63: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_61, add_64);  mul_61 = add_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_144: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_63, [4096, 4096]);  mul_63 = None
        permute_89: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg146_1, view_144, permute_89);  arg146_1 = view_144 = permute_89 = None
        view_145: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_53, [4, 1024, 1024]);  addmm_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_65: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_63, view_145);  add_63 = view_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_72: "f32[4, 1024, 1]" = var_mean_18[0]
        getitem_73: "f32[4, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        add_66: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_18: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_73);  add_65 = getitem_73 = None
        mul_64: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_65: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_64, arg147_1);  mul_64 = arg147_1 = None
        add_67: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_65, arg148_1);  mul_65 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_146: "f32[4096, 1024]" = torch.ops.aten.view.default(add_67, [4096, 1024])
        permute_90: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg150_1, view_146, permute_90);  arg150_1 = view_146 = permute_90 = None
        view_147: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_54, [4, 1024, 1024]);  addmm_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_148: "f32[4096, 1024]" = torch.ops.aten.view.default(add_67, [4096, 1024])
        permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg152_1, view_148, permute_91);  arg152_1 = view_148 = permute_91 = None
        view_149: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_55, [4, 1024, 1024]);  addmm_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_150: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_149, [4, -1, 16, 64]);  view_149 = None
        permute_92: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        clone_55: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_151: "f32[4096, 1024]" = torch.ops.aten.view.default(add_67, [4096, 1024])
        permute_93: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg154_1, view_151, permute_93);  arg154_1 = view_151 = permute_93 = None
        view_152: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_56, [4, 1024, 1024]);  addmm_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_153: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_152, [4, -1, 16, 64]);  view_152 = None
        permute_94: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        clone_56: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_154: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_147, [4, 1024, 16, 64]);  view_147 = None
        permute_95: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        clone_57: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_20: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_21: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 1);  unsqueeze_20 = None
        expand_30: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_21, [4, 1, 1024, 1024]);  unsqueeze_21 = None
        expand_31: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_30, [4, 16, 1024, 1024]);  expand_30 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_57, clone_55, clone_56, expand_31, False);  clone_57 = expand_31 = None
        getitem_74: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_96: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_155: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_96, [4, 1024, 1024]);  permute_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_156: "f32[4096, 1024]" = torch.ops.aten.view.default(view_155, [4096, 1024]);  view_155 = None
        permute_97: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg156_1, view_156, permute_97);  arg156_1 = view_156 = permute_97 = None
        view_157: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_57, [4, 1024, 1024]);  addmm_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_68: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_67, view_157);  add_67 = view_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_78: "f32[4, 1024, 1]" = var_mean_19[0]
        getitem_79: "f32[4, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        add_69: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_19: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_19: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_68, getitem_79);  add_68 = getitem_79 = None
        mul_66: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_67: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_66, arg157_1);  mul_66 = arg157_1 = None
        add_70: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_67, arg158_1);  mul_67 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_158: "f32[4096, 1024]" = torch.ops.aten.view.default(add_70, [4096, 1024])
        permute_98: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg160_1, view_158, permute_98);  arg160_1 = view_158 = permute_98 = None
        view_159: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_58, [4, 1024, 4096]);  addmm_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_68: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
        mul_69: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476);  view_159 = None
        erf_9: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_71: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_70: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_160: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_70, [4096, 4096]);  mul_70 = None
        permute_99: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg162_1, view_160, permute_99);  arg162_1 = view_160 = permute_99 = None
        view_161: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_59, [4, 1024, 1024]);  addmm_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_72: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_70, view_161);  add_70 = view_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_80: "f32[4, 1024, 1]" = var_mean_20[0]
        getitem_81: "f32[4, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        add_73: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_20: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_72, getitem_81);  add_72 = getitem_81 = None
        mul_71: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_72: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_71, arg163_1);  mul_71 = arg163_1 = None
        add_74: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_72, arg164_1);  mul_72 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_162: "f32[4096, 1024]" = torch.ops.aten.view.default(add_74, [4096, 1024])
        permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg166_1, view_162, permute_100);  arg166_1 = view_162 = permute_100 = None
        view_163: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_60, [4, 1024, 1024]);  addmm_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_164: "f32[4096, 1024]" = torch.ops.aten.view.default(add_74, [4096, 1024])
        permute_101: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg168_1, view_164, permute_101);  arg168_1 = view_164 = permute_101 = None
        view_165: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_61, [4, 1024, 1024]);  addmm_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_166: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_165, [4, -1, 16, 64]);  view_165 = None
        permute_102: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        clone_61: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_167: "f32[4096, 1024]" = torch.ops.aten.view.default(add_74, [4096, 1024])
        permute_103: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg170_1, view_167, permute_103);  arg170_1 = view_167 = permute_103 = None
        view_168: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_62, [4, 1024, 1024]);  addmm_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_169: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_168, [4, -1, 16, 64]);  view_168 = None
        permute_104: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        clone_62: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_170: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_163, [4, 1024, 16, 64]);  view_163 = None
        permute_105: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_63: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_22: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_23: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None
        expand_33: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_23, [4, 1, 1024, 1024]);  unsqueeze_23 = None
        expand_34: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_33, [4, 16, 1024, 1024]);  expand_33 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_63, clone_61, clone_62, expand_34, False);  clone_63 = expand_34 = None
        getitem_82: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_171: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_106, [4, 1024, 1024]);  permute_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_172: "f32[4096, 1024]" = torch.ops.aten.view.default(view_171, [4096, 1024]);  view_171 = None
        permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg172_1, view_172, permute_107);  arg172_1 = view_172 = permute_107 = None
        view_173: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_63, [4, 1024, 1024]);  addmm_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_75: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_74, view_173);  add_74 = view_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_86: "f32[4, 1024, 1]" = var_mean_21[0]
        getitem_87: "f32[4, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        add_76: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_21: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_21: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_75, getitem_87);  add_75 = getitem_87 = None
        mul_73: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_74: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg173_1);  mul_73 = arg173_1 = None
        add_77: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg174_1);  mul_74 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_174: "f32[4096, 1024]" = torch.ops.aten.view.default(add_77, [4096, 1024])
        permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg176_1, view_174, permute_108);  arg176_1 = view_174 = permute_108 = None
        view_175: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_64, [4, 1024, 4096]);  addmm_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_75: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_76: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_10: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_78: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_77: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_75, add_78);  mul_75 = add_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_176: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_77, [4096, 4096]);  mul_77 = None
        permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg178_1, view_176, permute_109);  arg178_1 = view_176 = permute_109 = None
        view_177: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_65, [4, 1024, 1024]);  addmm_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_79: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_77, view_177);  add_77 = view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
        getitem_88: "f32[4, 1024, 1]" = var_mean_22[0]
        getitem_89: "f32[4, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        add_80: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_22: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_79, getitem_89);  add_79 = getitem_89 = None
        mul_78: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_79: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_78, arg179_1);  mul_78 = arg179_1 = None
        add_81: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_79, arg180_1);  mul_79 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:450 in forward, code: query_states = self.q_proj(hidden_states)
        view_178: "f32[4096, 1024]" = torch.ops.aten.view.default(add_81, [4096, 1024])
        permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg182_1, view_178, permute_110);  arg182_1 = view_178 = permute_110 = None
        view_179: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_66, [4, 1024, 1024]);  addmm_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:475 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_180: "f32[4096, 1024]" = torch.ops.aten.view.default(add_81, [4096, 1024])
        permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg184_1, view_180, permute_111);  arg184_1 = view_180 = permute_111 = None
        view_181: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_67, [4, 1024, 1024]);  addmm_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_182: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_181, [4, -1, 16, 64]);  view_181 = None
        permute_112: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_67: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:476 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_183: "f32[4096, 1024]" = torch.ops.aten.view.default(add_81, [4096, 1024])
        permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg186_1, view_183, permute_113);  arg186_1 = view_183 = permute_113 = None
        view_184: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_68, [4, 1024, 1024]);  addmm_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_185: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_184, [4, -1, 16, 64]);  view_184 = None
        permute_114: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_68: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:167 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_186: "f32[4, 1024, 16, 64]" = torch.ops.aten.view.default(view_179, [4, 1024, 16, 64]);  view_179 = None
        permute_115: "f32[4, 16, 1024, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_69: "f32[4, 16, 1024, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:497 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_24: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_25: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 1);  unsqueeze_24 = None
        expand_36: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_25, [4, 1, 1024, 1024]);  unsqueeze_25 = None
        expand_37: "f32[4, 16, 1024, 1024]" = torch.ops.aten.expand.default(expand_36, [4, 16, 1024, 1024]);  expand_36 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_69, clone_67, clone_68, expand_37, False);  clone_69 = expand_37 = None
        getitem_90: "f32[4, 16, 1024, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:512 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_116: "f32[4, 1024, 16, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:516 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        view_187: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(permute_116, [4, 1024, 1024]);  permute_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:518 in forward, code: attn_output = self.out_proj(attn_output)
        view_188: "f32[4096, 1024]" = torch.ops.aten.view.default(view_187, [4096, 1024]);  view_187 = None
        permute_117: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg188_1, view_188, permute_117);  arg188_1 = view_188 = permute_117 = None
        view_189: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_69, [4, 1024, 1024]);  addmm_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:674 in forward, code: hidden_states = residual + hidden_states
        add_82: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_81, view_189);  add_81 = view_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:675 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
        getitem_94: "f32[4, 1024, 1]" = var_mean_23[0]
        getitem_95: "f32[4, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        add_83: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_23: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        sub_23: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_82, getitem_95);  add_82 = getitem_95 = None
        mul_80: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_81: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_80, arg189_1);  mul_80 = arg189_1 = None
        add_84: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_81, arg190_1);  mul_81 = arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:702 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_190: "f32[4096, 1024]" = torch.ops.aten.view.default(add_84, [4096, 1024])
        permute_118: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "f32[4096, 4096]" = torch.ops.aten.addmm.default(arg192_1, view_190, permute_118);  arg192_1 = view_190 = permute_118 = None
        view_191: "f32[4, 1024, 4096]" = torch.ops.aten.view.default(addmm_70, [4, 1024, 4096]);  addmm_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_82: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_191, 0.5)
        mul_83: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476);  view_191 = None
        erf_11: "f32[4, 1024, 4096]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_85: "f32[4, 1024, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_84: "f32[4, 1024, 4096]" = torch.ops.aten.mul.Tensor(mul_82, add_85);  mul_82 = add_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:704 in forward, code: hidden_states = self.fc2(hidden_states)
        view_192: "f32[4096, 4096]" = torch.ops.aten.view.default(mul_84, [4096, 4096]);  mul_84 = None
        permute_119: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "f32[4096, 1024]" = torch.ops.aten.addmm.default(arg194_1, view_192, permute_119);  arg194_1 = view_192 = permute_119 = None
        view_193: "f32[4, 1024, 1024]" = torch.ops.aten.view.default(addmm_71, [4, 1024, 1024]);  addmm_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:706 in forward, code: hidden_states = residual + hidden_states
        add_86: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(add_84, view_193);  add_84 = view_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:707 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
        getitem_96: "f32[4, 1024, 1]" = var_mean_24[0]
        getitem_97: "f32[4, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
        add_87: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_24: "f32[4, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_86, getitem_97);  add_86 = getitem_97 = None
        mul_85: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_86: "f32[4, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_85, arg195_1);  mul_85 = arg195_1 = None
        add_88: "f32[4, 1024, 1024]" = torch.ops.aten.add.Tensor(mul_86, arg196_1);  mul_86 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:2171 in forward, code: logits = self.lm_head(outputs[0])
        permute_120: "f32[1024, 50265]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_194: "f32[4096, 1024]" = torch.ops.aten.view.default(add_88, [4096, 1024]);  add_88 = None
        
        # No stacktrace found for following nodes
        full_default_4: "f32[1024, 3]" = torch.ops.aten.full.default([1024, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[1024, 50268]" = torch.ops.aten.cat.default([permute_120, full_default_4], 1);  permute_120 = full_default_4 = None
        mm_default: "f32[4096, 50268]" = torch.ops.aten.mm.default(view_194, cat_default);  view_194 = cat_default = None
        slice_tensor: "f32[4096, 50265]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:2171 in forward, code: logits = self.lm_head(outputs[0])
        view_195: "f32[4, 1024, 50265]" = torch.ops.aten.view.default(slice_tensor, [4, 1024, 50265]);  slice_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/bart/modeling_bart.py:2177 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_196: "f32[4096, 50265]" = torch.ops.aten.view.default(view_195, [-1, 50265])
        view_197: "i64[4096]" = torch.ops.aten.view.default(arg197_1, [-1]);  arg197_1 = None
        amax: "f32[4096, 1]" = torch.ops.aten.amax.default(view_196, [1], True)
        sub_25: "f32[4096, 50265]" = torch.ops.aten.sub.Tensor(view_196, amax);  view_196 = amax = None
        exp: "f32[4096, 50265]" = torch.ops.aten.exp.default(sub_25)
        sum_1: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_26: "f32[4096, 50265]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_197, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[4096]" = torch.ops.aten.where.self(ne, view_197, full_default_2);  ne = full_default_2 = None
        unsqueeze_26: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_26, 1, unsqueeze_26);  sub_26 = unsqueeze_26 = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1: "b8[4096]" = torch.ops.aten.ne.Scalar(view_197, -100)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2: "b8[4096]" = torch.ops.aten.ne.Scalar(view_197, -100);  view_197 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None
        return (div, view_195, clone_1, clone_2, clone_7, clone_8, clone_13, clone_14, clone_19, clone_20, clone_25, clone_26, clone_31, clone_32, clone_37, clone_38, clone_43, clone_44, clone_49, clone_50, clone_55, clone_56, clone_61, clone_62, clone_67, clone_68)
        