class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 1024]", arg1_1: "f32[50257, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[2304]", arg6_1: "f32[768, 2304]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[2304]", arg18_1: "f32[768, 2304]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[768]", arg26_1: "f32[3072, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[2304]", arg30_1: "f32[768, 2304]", arg31_1: "f32[768]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[3072, 768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[2304]", arg42_1: "f32[768, 2304]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[3072, 768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[2304]", arg54_1: "f32[768, 2304]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[2304]", arg66_1: "f32[768, 2304]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[3072, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[2304]", arg78_1: "f32[768, 2304]", arg79_1: "f32[768]", arg80_1: "f32[768, 768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072]", arg84_1: "f32[768, 3072]", arg85_1: "f32[768]", arg86_1: "f32[3072, 768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[2304]", arg90_1: "f32[768, 2304]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[3072, 768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[2304]", arg102_1: "f32[768, 2304]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[3072]", arg108_1: "f32[768, 3072]", arg109_1: "f32[768]", arg110_1: "f32[3072, 768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[2304]", arg114_1: "f32[768, 2304]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[3072]", arg120_1: "f32[768, 3072]", arg121_1: "f32[768]", arg122_1: "f32[3072, 768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[2304]", arg126_1: "f32[768, 2304]", arg127_1: "f32[768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[3072, 768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[2304]", arg138_1: "f32[768, 2304]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[3072, 768]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[2, 768]", arg150_1: "i64[4]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1005 in forward, code: input_ids = input_ids.view(-1, input_shape[-1])
        view: "i64[4, 1024]" = torch.ops.aten.view.default(arg0_1, [-1, 1024])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1024 in forward, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1025 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1028 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[4, 1024, 768]" = torch.ops.aten.embedding.default(arg1_1, view);  arg1_1 = view = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1029 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 1024, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = unsqueeze = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1030 in forward, code: hidden_states = inputs_embeds + position_embeds
        add: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:158 in _make_causal_mask, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        full_default: "f32[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:159 in _make_causal_mask, code: mask_cond = torch.arange(mask.size(-1), device=device)
        iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:160 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        add_1: "i64[1024]" = torch.ops.aten.add.Tensor(iota_1, 1)
        view_1: "i64[1024, 1]" = torch.ops.aten.view.default(add_1, [1024, 1]);  add_1 = None
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Tensor(iota_1, view_1);  iota_1 = view_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1024, 1024]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 1024, 1]" = var_mean[0]
        getitem_1: "f32[4, 1024, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_3: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_2: "f32[4096, 768]" = torch.ops.aten.view.default(add_3, [-1, 768]);  add_3 = None
        addmm: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg5_1, view_2, arg6_1);  arg5_1 = view_2 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_3: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm, [4, 1024, 2304]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_3, 768, 2);  view_3 = None
        getitem_2: "f32[4, 1024, 768]" = split[0]
        getitem_3: "f32[4, 1024, 768]" = split[1]
        getitem_4: "f32[4, 1024, 768]" = split[2];  split = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_4: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_2, [4, 1024, 12, 64]);  getitem_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_5: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_3, [4, 1024, 12, 64]);  getitem_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_6: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_4, [4, 1024, 12, 64]);  getitem_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_2: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_3: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_4: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        expand_2: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_4, [4, 1, 1024, 1024]);  unsqueeze_4 = None
        expand_3: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_2, [4, 12, 1024, 1024]);  expand_2 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute, permute_1, permute_2, expand_3, False);  permute = expand_3 = None
        getitem_5: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_7: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_3, [4, 1024, 768]);  permute_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_8: "f32[4096, 768]" = torch.ops.aten.view.default(view_7, [-1, 768]);  view_7 = None
        addmm_1: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg7_1, view_8, arg8_1);  arg7_1 = view_8 = arg8_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_9: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_1, [4, 1024, 768]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_4: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_9, add);  view_9 = add = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_9: "f32[4, 1024, 1]" = var_mean_1[0]
        getitem_10: "f32[4, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_10);  getitem_10 = None
        mul_2: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_6: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_10: "f32[4096, 768]" = torch.ops.aten.view.default(add_6, [-1, 768]);  add_6 = None
        addmm_2: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg11_1, view_10, arg12_1);  arg11_1 = view_10 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_11: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_2, [4, 1024, 3072]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
        pow_1: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
        mul_5: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_11, mul_5);  view_11 = mul_5 = None
        mul_6: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_12: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg13_1, view_12, arg14_1);  arg13_1 = view_12 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_13: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_3, [4, 1024, 768]);  addmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_9: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_4, view_13);  add_4 = view_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_11: "f32[4, 1024, 1]" = var_mean_2[0]
        getitem_12: "f32[4, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_12);  getitem_12 = None
        mul_8: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_11: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_14: "f32[4096, 768]" = torch.ops.aten.view.default(add_11, [-1, 768]);  add_11 = None
        addmm_4: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg17_1, view_14, arg18_1);  arg17_1 = view_14 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_15: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_4, [4, 1024, 2304]);  addmm_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_15, 768, 2);  view_15 = None
        getitem_13: "f32[4, 1024, 768]" = split_1[0]
        getitem_14: "f32[4, 1024, 768]" = split_1[1]
        getitem_15: "f32[4, 1024, 768]" = split_1[2];  split_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_16: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_13, [4, 1024, 12, 64]);  getitem_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_4: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_17: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_14, [4, 1024, 12, 64]);  getitem_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_5: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_18: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_15, [4, 1024, 12, 64]);  getitem_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_6: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_5: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_6: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
        expand_5: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_6, [4, 1, 1024, 1024]);  unsqueeze_6 = None
        expand_6: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_5, [4, 12, 1024, 1024]);  expand_5 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_4, permute_5, permute_6, expand_6, False);  permute_4 = expand_6 = None
        getitem_16: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_19: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_7, [4, 1024, 768]);  permute_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_20: "f32[4096, 768]" = torch.ops.aten.view.default(view_19, [-1, 768]);  view_19 = None
        addmm_5: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg19_1, view_20, arg20_1);  arg19_1 = view_20 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_21: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_5, [4, 1024, 768]);  addmm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_12: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_21, add_9);  view_21 = add_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_20: "f32[4, 1024, 1]" = var_mean_3[0]
        getitem_21: "f32[4, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_21);  getitem_21 = None
        mul_10: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_14: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_22: "f32[4096, 768]" = torch.ops.aten.view.default(add_14, [-1, 768]);  add_14 = None
        addmm_6: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg23_1, view_22, arg24_1);  arg23_1 = view_22 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_23: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_6, [4, 1024, 3072]);  addmm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
        pow_2: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
        mul_13: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_23, mul_13);  view_23 = mul_13 = None
        mul_14: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_24: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg25_1, view_24, arg26_1);  arg25_1 = view_24 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_25: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_7, [4, 1024, 768]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_17: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_12, view_25);  add_12 = view_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22: "f32[4, 1024, 1]" = var_mean_4[0]
        getitem_23: "f32[4, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  getitem_23 = None
        mul_16: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_19: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_26: "f32[4096, 768]" = torch.ops.aten.view.default(add_19, [-1, 768]);  add_19 = None
        addmm_8: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg29_1, view_26, arg30_1);  arg29_1 = view_26 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_27: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_8, [4, 1024, 2304]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_27, 768, 2);  view_27 = None
        getitem_24: "f32[4, 1024, 768]" = split_2[0]
        getitem_25: "f32[4, 1024, 768]" = split_2[1]
        getitem_26: "f32[4, 1024, 768]" = split_2[2];  split_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_28: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_24, [4, 1024, 12, 64]);  getitem_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_8: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_29: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_25, [4, 1024, 12, 64]);  getitem_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_9: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_30: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_26, [4, 1024, 12, 64]);  getitem_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_10: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_7: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_8: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        expand_8: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_8, [4, 1, 1024, 1024]);  unsqueeze_8 = None
        expand_9: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_8, [4, 12, 1024, 1024]);  expand_8 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_8, permute_9, permute_10, expand_9, False);  permute_8 = expand_9 = None
        getitem_27: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_31: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_11, [4, 1024, 768]);  permute_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_32: "f32[4096, 768]" = torch.ops.aten.view.default(view_31, [-1, 768]);  view_31 = None
        addmm_9: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg31_1, view_32, arg32_1);  arg31_1 = view_32 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_33: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_9, [4, 1024, 768]);  addmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_20: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_33, add_17);  view_33 = add_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_31: "f32[4, 1024, 1]" = var_mean_5[0]
        getitem_32: "f32[4, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_32);  getitem_32 = None
        mul_18: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_19: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_22: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_34: "f32[4096, 768]" = torch.ops.aten.view.default(add_22, [-1, 768]);  add_22 = None
        addmm_10: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg35_1, view_34, arg36_1);  arg35_1 = view_34 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_35: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, [4, 1024, 3072]);  addmm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
        pow_3: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
        mul_21: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_21);  view_35 = mul_21 = None
        mul_22: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_36: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg37_1, view_36, arg38_1);  arg37_1 = view_36 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_37: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_11, [4, 1024, 768]);  addmm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_25: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_20, view_37);  add_20 = view_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_33: "f32[4, 1024, 1]" = var_mean_6[0]
        getitem_34: "f32[4, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_6: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_34);  getitem_34 = None
        mul_24: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_25: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_27: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_38: "f32[4096, 768]" = torch.ops.aten.view.default(add_27, [-1, 768]);  add_27 = None
        addmm_12: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg41_1, view_38, arg42_1);  arg41_1 = view_38 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_39: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_12, [4, 1024, 2304]);  addmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_39, 768, 2);  view_39 = None
        getitem_35: "f32[4, 1024, 768]" = split_3[0]
        getitem_36: "f32[4, 1024, 768]" = split_3[1]
        getitem_37: "f32[4, 1024, 768]" = split_3[2];  split_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_40: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_35, [4, 1024, 12, 64]);  getitem_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_12: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_41: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_36, [4, 1024, 12, 64]);  getitem_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_13: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_42: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_37, [4, 1024, 12, 64]);  getitem_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_14: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_9: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_10: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        expand_11: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_10, [4, 1, 1024, 1024]);  unsqueeze_10 = None
        expand_12: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_11, [4, 12, 1024, 1024]);  expand_11 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_12, permute_13, permute_14, expand_12, False);  permute_12 = expand_12 = None
        getitem_38: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_43: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_15, [4, 1024, 768]);  permute_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_44: "f32[4096, 768]" = torch.ops.aten.view.default(view_43, [-1, 768]);  view_43 = None
        addmm_13: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg43_1, view_44, arg44_1);  arg43_1 = view_44 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_45: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_13, [4, 1024, 768]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_28: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_45, add_25);  view_45 = add_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_42: "f32[4, 1024, 1]" = var_mean_7[0]
        getitem_43: "f32[4, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_43);  getitem_43 = None
        mul_26: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_27: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_30: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_46: "f32[4096, 768]" = torch.ops.aten.view.default(add_30, [-1, 768]);  add_30 = None
        addmm_14: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg47_1, view_46, arg48_1);  arg47_1 = view_46 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_47: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_14, [4, 1024, 3072]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
        pow_4: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
        mul_29: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_47, mul_29);  view_47 = mul_29 = None
        mul_30: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_48: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg49_1, view_48, arg50_1);  arg49_1 = view_48 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_49: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_15, [4, 1024, 768]);  addmm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_33: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_28, view_49);  add_28 = view_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_44: "f32[4, 1024, 1]" = var_mean_8[0]
        getitem_45: "f32[4, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_8: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_45);  getitem_45 = None
        mul_32: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_33: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_35: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_50: "f32[4096, 768]" = torch.ops.aten.view.default(add_35, [-1, 768]);  add_35 = None
        addmm_16: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg53_1, view_50, arg54_1);  arg53_1 = view_50 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_51: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_16, [4, 1024, 2304]);  addmm_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_51, 768, 2);  view_51 = None
        getitem_46: "f32[4, 1024, 768]" = split_4[0]
        getitem_47: "f32[4, 1024, 768]" = split_4[1]
        getitem_48: "f32[4, 1024, 768]" = split_4[2];  split_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_52: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_46, [4, 1024, 12, 64]);  getitem_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_16: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_53: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_47, [4, 1024, 12, 64]);  getitem_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_17: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_54: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_48, [4, 1024, 12, 64]);  getitem_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_18: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_11: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_12: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 1);  unsqueeze_11 = None
        expand_14: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_12, [4, 1, 1024, 1024]);  unsqueeze_12 = None
        expand_15: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_14, [4, 12, 1024, 1024]);  expand_14 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_16, permute_17, permute_18, expand_15, False);  permute_16 = expand_15 = None
        getitem_49: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_55: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_19, [4, 1024, 768]);  permute_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_56: "f32[4096, 768]" = torch.ops.aten.view.default(view_55, [-1, 768]);  view_55 = None
        addmm_17: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg55_1, view_56, arg56_1);  arg55_1 = view_56 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_57: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_17, [4, 1024, 768]);  addmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_36: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_57, add_33);  view_57 = add_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_53: "f32[4, 1024, 1]" = var_mean_9[0]
        getitem_54: "f32[4, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        add_37: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_54);  getitem_54 = None
        mul_34: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_35: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_38: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_58: "f32[4096, 768]" = torch.ops.aten.view.default(add_38, [-1, 768]);  add_38 = None
        addmm_18: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg59_1, view_58, arg60_1);  arg59_1 = view_58 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_59: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_18, [4, 1024, 3072]);  addmm_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_59, 0.5)
        pow_5: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_59, 3.0)
        mul_37: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_59, mul_37);  view_59 = mul_37 = None
        mul_38: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_60: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg61_1, view_60, arg62_1);  arg61_1 = view_60 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_61: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_19, [4, 1024, 768]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_41: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_36, view_61);  add_36 = view_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_55: "f32[4, 1024, 1]" = var_mean_10[0]
        getitem_56: "f32[4, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_10: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_56);  getitem_56 = None
        mul_40: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_41: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_43: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_62: "f32[4096, 768]" = torch.ops.aten.view.default(add_43, [-1, 768]);  add_43 = None
        addmm_20: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg65_1, view_62, arg66_1);  arg65_1 = view_62 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_63: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_20, [4, 1024, 2304]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_63, 768, 2);  view_63 = None
        getitem_57: "f32[4, 1024, 768]" = split_5[0]
        getitem_58: "f32[4, 1024, 768]" = split_5[1]
        getitem_59: "f32[4, 1024, 768]" = split_5[2];  split_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_64: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_57, [4, 1024, 12, 64]);  getitem_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_20: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_65: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_58, [4, 1024, 12, 64]);  getitem_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_21: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_66: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_59, [4, 1024, 12, 64]);  getitem_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_22: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_13: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_14: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None
        expand_17: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_14, [4, 1, 1024, 1024]);  unsqueeze_14 = None
        expand_18: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_17, [4, 12, 1024, 1024]);  expand_17 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_20, permute_21, permute_22, expand_18, False);  permute_20 = expand_18 = None
        getitem_60: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_67: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_23, [4, 1024, 768]);  permute_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_68: "f32[4096, 768]" = torch.ops.aten.view.default(view_67, [-1, 768]);  view_67 = None
        addmm_21: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg67_1, view_68, arg68_1);  arg67_1 = view_68 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_69: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_21, [4, 1024, 768]);  addmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_44: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_69, add_41);  view_69 = add_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_64: "f32[4, 1024, 1]" = var_mean_11[0]
        getitem_65: "f32[4, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        add_45: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_65);  getitem_65 = None
        mul_42: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_43: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_46: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_70: "f32[4096, 768]" = torch.ops.aten.view.default(add_46, [-1, 768]);  add_46 = None
        addmm_22: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg71_1, view_70, arg72_1);  arg71_1 = view_70 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_71: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, [4, 1024, 3072]);  addmm_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
        pow_6: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
        mul_45: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_45);  view_71 = mul_45 = None
        mul_46: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_72: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg73_1, view_72, arg74_1);  arg73_1 = view_72 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_73: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_23, [4, 1024, 768]);  addmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_49: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_44, view_73);  add_44 = view_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_66: "f32[4, 1024, 1]" = var_mean_12[0]
        getitem_67: "f32[4, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_67);  getitem_67 = None
        mul_48: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_51: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_74: "f32[4096, 768]" = torch.ops.aten.view.default(add_51, [-1, 768]);  add_51 = None
        addmm_24: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg77_1, view_74, arg78_1);  arg77_1 = view_74 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_75: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_24, [4, 1024, 2304]);  addmm_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_75, 768, 2);  view_75 = None
        getitem_68: "f32[4, 1024, 768]" = split_6[0]
        getitem_69: "f32[4, 1024, 768]" = split_6[1]
        getitem_70: "f32[4, 1024, 768]" = split_6[2];  split_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_76: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_68, [4, 1024, 12, 64]);  getitem_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_24: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_77: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_69, [4, 1024, 12, 64]);  getitem_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_25: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_78: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_70, [4, 1024, 12, 64]);  getitem_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_26: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_15: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_16: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 1);  unsqueeze_15 = None
        expand_20: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_16, [4, 1, 1024, 1024]);  unsqueeze_16 = None
        expand_21: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_20, [4, 12, 1024, 1024]);  expand_20 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_24, permute_25, permute_26, expand_21, False);  permute_24 = expand_21 = None
        getitem_71: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_79: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_27, [4, 1024, 768]);  permute_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_80: "f32[4096, 768]" = torch.ops.aten.view.default(view_79, [-1, 768]);  view_79 = None
        addmm_25: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg79_1, view_80, arg80_1);  arg79_1 = view_80 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_81: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_25, [4, 1024, 768]);  addmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_52: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_81, add_49);  view_81 = add_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_75: "f32[4, 1024, 1]" = var_mean_13[0]
        getitem_76: "f32[4, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        add_53: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_13: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_76);  getitem_76 = None
        mul_50: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_51: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg81_1);  mul_50 = arg81_1 = None
        add_54: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_51, arg82_1);  mul_51 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_82: "f32[4096, 768]" = torch.ops.aten.view.default(add_54, [-1, 768]);  add_54 = None
        addmm_26: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg83_1, view_82, arg84_1);  arg83_1 = view_82 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_83: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_26, [4, 1024, 3072]);  addmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_83, 0.5)
        pow_7: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_83, 3.0)
        mul_53: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_83, mul_53);  view_83 = mul_53 = None
        mul_54: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_56: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_84: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
        addmm_27: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg85_1, view_84, arg86_1);  arg85_1 = view_84 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_85: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_27, [4, 1024, 768]);  addmm_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_57: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_52, view_85);  add_52 = view_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_77: "f32[4, 1024, 1]" = var_mean_14[0]
        getitem_78: "f32[4, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        add_58: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_14: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_78);  getitem_78 = None
        mul_56: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_57: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg87_1);  mul_56 = arg87_1 = None
        add_59: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_57, arg88_1);  mul_57 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_86: "f32[4096, 768]" = torch.ops.aten.view.default(add_59, [-1, 768]);  add_59 = None
        addmm_28: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg89_1, view_86, arg90_1);  arg89_1 = view_86 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_87: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_28, [4, 1024, 2304]);  addmm_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_87, 768, 2);  view_87 = None
        getitem_79: "f32[4, 1024, 768]" = split_7[0]
        getitem_80: "f32[4, 1024, 768]" = split_7[1]
        getitem_81: "f32[4, 1024, 768]" = split_7[2];  split_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_88: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_79, [4, 1024, 12, 64]);  getitem_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_28: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_89: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_80, [4, 1024, 12, 64]);  getitem_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_29: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_90: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_81, [4, 1024, 12, 64]);  getitem_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_30: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_17: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_18: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 1);  unsqueeze_17 = None
        expand_23: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_18, [4, 1, 1024, 1024]);  unsqueeze_18 = None
        expand_24: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_23, [4, 12, 1024, 1024]);  expand_23 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_28, permute_29, permute_30, expand_24, False);  permute_28 = expand_24 = None
        getitem_82: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_91: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_31, [4, 1024, 768]);  permute_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_92: "f32[4096, 768]" = torch.ops.aten.view.default(view_91, [-1, 768]);  view_91 = None
        addmm_29: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg91_1, view_92, arg92_1);  arg91_1 = view_92 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_93: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_29, [4, 1024, 768]);  addmm_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_60: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_93, add_57);  view_93 = add_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_86: "f32[4, 1024, 1]" = var_mean_15[0]
        getitem_87: "f32[4, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        add_61: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_15: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_87);  getitem_87 = None
        mul_58: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_59: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg93_1);  mul_58 = arg93_1 = None
        add_62: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_59, arg94_1);  mul_59 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_94: "f32[4096, 768]" = torch.ops.aten.view.default(add_62, [-1, 768]);  add_62 = None
        addmm_30: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg95_1, view_94, arg96_1);  arg95_1 = view_94 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_95: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_30, [4, 1024, 3072]);  addmm_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_95, 0.5)
        pow_8: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_95, 3.0)
        mul_61: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_95, mul_61);  view_95 = mul_61 = None
        mul_62: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_64: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_96: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
        addmm_31: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg97_1, view_96, arg98_1);  arg97_1 = view_96 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_97: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_31, [4, 1024, 768]);  addmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_65: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_60, view_97);  add_60 = view_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_88: "f32[4, 1024, 1]" = var_mean_16[0]
        getitem_89: "f32[4, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        add_66: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_16: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_89);  getitem_89 = None
        mul_64: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_65: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg99_1);  mul_64 = arg99_1 = None
        add_67: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_98: "f32[4096, 768]" = torch.ops.aten.view.default(add_67, [-1, 768]);  add_67 = None
        addmm_32: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg101_1, view_98, arg102_1);  arg101_1 = view_98 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_99: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_32, [4, 1024, 2304]);  addmm_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_99, 768, 2);  view_99 = None
        getitem_90: "f32[4, 1024, 768]" = split_8[0]
        getitem_91: "f32[4, 1024, 768]" = split_8[1]
        getitem_92: "f32[4, 1024, 768]" = split_8[2];  split_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_100: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_90, [4, 1024, 12, 64]);  getitem_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_32: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_101: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_91, [4, 1024, 12, 64]);  getitem_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_33: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_102: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_92, [4, 1024, 12, 64]);  getitem_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_34: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_19: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_20: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 1);  unsqueeze_19 = None
        expand_26: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_20, [4, 1, 1024, 1024]);  unsqueeze_20 = None
        expand_27: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_26, [4, 12, 1024, 1024]);  expand_26 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_32, permute_33, permute_34, expand_27, False);  permute_32 = expand_27 = None
        getitem_93: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_103: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_35, [4, 1024, 768]);  permute_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_104: "f32[4096, 768]" = torch.ops.aten.view.default(view_103, [-1, 768]);  view_103 = None
        addmm_33: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg103_1, view_104, arg104_1);  arg103_1 = view_104 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_105: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_33, [4, 1024, 768]);  addmm_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_68: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_105, add_65);  view_105 = add_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_97: "f32[4, 1024, 1]" = var_mean_17[0]
        getitem_98: "f32[4, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        add_69: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_17: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_98);  getitem_98 = None
        mul_66: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_67: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg105_1);  mul_66 = arg105_1 = None
        add_70: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_67, arg106_1);  mul_67 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_106: "f32[4096, 768]" = torch.ops.aten.view.default(add_70, [-1, 768]);  add_70 = None
        addmm_34: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg107_1, view_106, arg108_1);  arg107_1 = view_106 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_107: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_34, [4, 1024, 3072]);  addmm_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        pow_9: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
        mul_69: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_107, mul_69);  view_107 = mul_69 = None
        mul_70: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_72: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_108: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_35: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg109_1, view_108, arg110_1);  arg109_1 = view_108 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_109: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_35, [4, 1024, 768]);  addmm_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_73: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_68, view_109);  add_68 = view_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_99: "f32[4, 1024, 1]" = var_mean_18[0]
        getitem_100: "f32[4, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        add_74: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_18: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_100);  getitem_100 = None
        mul_72: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_73: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg111_1);  mul_72 = arg111_1 = None
        add_75: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_73, arg112_1);  mul_73 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_110: "f32[4096, 768]" = torch.ops.aten.view.default(add_75, [-1, 768]);  add_75 = None
        addmm_36: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg113_1, view_110, arg114_1);  arg113_1 = view_110 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_111: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_36, [4, 1024, 2304]);  addmm_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_111, 768, 2);  view_111 = None
        getitem_101: "f32[4, 1024, 768]" = split_9[0]
        getitem_102: "f32[4, 1024, 768]" = split_9[1]
        getitem_103: "f32[4, 1024, 768]" = split_9[2];  split_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_112: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_101, [4, 1024, 12, 64]);  getitem_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_36: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_113: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_102, [4, 1024, 12, 64]);  getitem_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_37: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_114: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_103, [4, 1024, 12, 64]);  getitem_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_38: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_21: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_22: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 1);  unsqueeze_21 = None
        expand_29: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_22, [4, 1, 1024, 1024]);  unsqueeze_22 = None
        expand_30: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_29, [4, 12, 1024, 1024]);  expand_29 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_36, permute_37, permute_38, expand_30, False);  permute_36 = expand_30 = None
        getitem_104: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_115: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_39, [4, 1024, 768]);  permute_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_116: "f32[4096, 768]" = torch.ops.aten.view.default(view_115, [-1, 768]);  view_115 = None
        addmm_37: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg115_1, view_116, arg116_1);  arg115_1 = view_116 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_117: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_37, [4, 1024, 768]);  addmm_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_76: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_117, add_73);  view_117 = add_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_108: "f32[4, 1024, 1]" = var_mean_19[0]
        getitem_109: "f32[4, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        add_77: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_19: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_109);  getitem_109 = None
        mul_74: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_75: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg117_1);  mul_74 = arg117_1 = None
        add_78: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_75, arg118_1);  mul_75 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_118: "f32[4096, 768]" = torch.ops.aten.view.default(add_78, [-1, 768]);  add_78 = None
        addmm_38: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg119_1, view_118, arg120_1);  arg119_1 = view_118 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_119: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_38, [4, 1024, 3072]);  addmm_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_119, 0.5)
        pow_10: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_119, 3.0)
        mul_77: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_119, mul_77);  view_119 = mul_77 = None
        mul_78: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_80: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_120: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
        addmm_39: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg121_1, view_120, arg122_1);  arg121_1 = view_120 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_121: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_39, [4, 1024, 768]);  addmm_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_81: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_76, view_121);  add_76 = view_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_110: "f32[4, 1024, 1]" = var_mean_20[0]
        getitem_111: "f32[4, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        add_82: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_20: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_111);  getitem_111 = None
        mul_80: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_81: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg123_1);  mul_80 = arg123_1 = None
        add_83: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_81, arg124_1);  mul_81 = arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_122: "f32[4096, 768]" = torch.ops.aten.view.default(add_83, [-1, 768]);  add_83 = None
        addmm_40: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg125_1, view_122, arg126_1);  arg125_1 = view_122 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_123: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_40, [4, 1024, 2304]);  addmm_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_123, 768, 2);  view_123 = None
        getitem_112: "f32[4, 1024, 768]" = split_10[0]
        getitem_113: "f32[4, 1024, 768]" = split_10[1]
        getitem_114: "f32[4, 1024, 768]" = split_10[2];  split_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_124: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_112, [4, 1024, 12, 64]);  getitem_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_40: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_125: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_113, [4, 1024, 12, 64]);  getitem_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_41: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_126: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_114, [4, 1024, 12, 64]);  getitem_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_42: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_23: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_24: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 1);  unsqueeze_23 = None
        expand_32: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_24, [4, 1, 1024, 1024]);  unsqueeze_24 = None
        expand_33: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_32, [4, 12, 1024, 1024]);  expand_32 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_40, permute_41, permute_42, expand_33, False);  permute_40 = expand_33 = None
        getitem_115: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_127: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_43, [4, 1024, 768]);  permute_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_128: "f32[4096, 768]" = torch.ops.aten.view.default(view_127, [-1, 768]);  view_127 = None
        addmm_41: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg127_1, view_128, arg128_1);  arg127_1 = view_128 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_129: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_41, [4, 1024, 768]);  addmm_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_84: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_129, add_81);  view_129 = add_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_119: "f32[4, 1024, 1]" = var_mean_21[0]
        getitem_120: "f32[4, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        add_85: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_21: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_120);  getitem_120 = None
        mul_82: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_83: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg129_1);  mul_82 = arg129_1 = None
        add_86: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_83, arg130_1);  mul_83 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_130: "f32[4096, 768]" = torch.ops.aten.view.default(add_86, [-1, 768]);  add_86 = None
        addmm_42: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_130, arg132_1);  arg131_1 = view_130 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_131: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_42, [4, 1024, 3072]);  addmm_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        pow_11: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
        mul_85: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_131, mul_85);  view_131 = mul_85 = None
        mul_86: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_88: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_132: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
        addmm_43: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg133_1, view_132, arg134_1);  arg133_1 = view_132 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_133: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_43, [4, 1024, 768]);  addmm_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_89: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_84, view_133);  add_84 = view_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_121: "f32[4, 1024, 1]" = var_mean_22[0]
        getitem_122: "f32[4, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        add_90: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_22: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_122);  getitem_122 = None
        mul_88: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_89: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg135_1);  mul_88 = arg135_1 = None
        add_91: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, arg136_1);  mul_89 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_134: "f32[4096, 768]" = torch.ops.aten.view.default(add_91, [-1, 768]);  add_91 = None
        addmm_44: "f32[4096, 2304]" = torch.ops.aten.addmm.default(arg137_1, view_134, arg138_1);  arg137_1 = view_134 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_135: "f32[4, 1024, 2304]" = torch.ops.aten.view.default(addmm_44, [4, 1024, 2304]);  addmm_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_135, 768, 2);  view_135 = None
        getitem_123: "f32[4, 1024, 768]" = split_11[0]
        getitem_124: "f32[4, 1024, 768]" = split_11[1]
        getitem_125: "f32[4, 1024, 768]" = split_11[2];  split_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_136: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_123, [4, 1024, 12, 64]);  getitem_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_44: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_137: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_124, [4, 1024, 12, 64]);  getitem_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_45: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_138: "f32[4, 1024, 12, 64]" = torch.ops.aten.view.default(getitem_125, [4, 1024, 12, 64]);  getitem_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_46: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_25: "f32[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_26: "f32[1, 1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 1);  unsqueeze_25 = None
        expand_35: "f32[4, 1, 1024, 1024]" = torch.ops.aten.expand.default(unsqueeze_26, [4, 1, 1024, 1024]);  unsqueeze_26 = None
        expand_36: "f32[4, 12, 1024, 1024]" = torch.ops.aten.expand.default(expand_35, [4, 12, 1024, 1024]);  expand_35 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_44, permute_45, permute_46, expand_36, False);  permute_44 = expand_36 = None
        getitem_126: "f32[4, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_139: "f32[4, 1024, 768]" = torch.ops.aten.view.default(permute_47, [4, 1024, 768]);  permute_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_140: "f32[4096, 768]" = torch.ops.aten.view.default(view_139, [-1, 768]);  view_139 = None
        addmm_45: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg139_1, view_140, arg140_1);  arg139_1 = view_140 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_141: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_45, [4, 1024, 768]);  addmm_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_92: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_141, add_89);  view_141 = add_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_130: "f32[4, 1024, 1]" = var_mean_23[0]
        getitem_131: "f32[4, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        add_93: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_23: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_131);  getitem_131 = None
        mul_90: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_91: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg141_1);  mul_90 = arg141_1 = None
        add_94: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, arg142_1);  mul_91 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_142: "f32[4096, 768]" = torch.ops.aten.view.default(add_94, [-1, 768]);  add_94 = None
        addmm_46: "f32[4096, 3072]" = torch.ops.aten.addmm.default(arg143_1, view_142, arg144_1);  arg143_1 = view_142 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_143: "f32[4, 1024, 3072]" = torch.ops.aten.view.default(addmm_46, [4, 1024, 3072]);  addmm_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
        pow_12: "f32[4, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_143, 3.0)
        mul_93: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(view_143, mul_93);  view_143 = mul_93 = None
        mul_94: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11: "f32[4, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_96: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_144: "f32[4096, 3072]" = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_47: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg145_1, view_144, arg146_1);  arg145_1 = view_144 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_145: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm_47, [4, 1024, 768]);  addmm_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_97: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_92, view_145);  add_92 = view_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1156 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_132: "f32[4, 1024, 1]" = var_mean_24[0]
        getitem_133: "f32[4, 1024, 1]" = var_mean_24[1];  var_mean_24 = None
        add_98: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_24: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_133);  add_97 = getitem_133 = None
        mul_96: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_97: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg147_1);  mul_96 = arg147_1 = None
        add_99: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, arg148_1);  mul_97 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1158 in forward, code: hidden_states = hidden_states.view(output_shape)
        view_146: "f32[4, 1024, 768]" = torch.ops.aten.view.default(add_99, [-1, 1024, 768]);  add_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1690 in forward, code: logits = self.score(hidden_states)
        permute_48: "f32[768, 2]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        view_147: "f32[4096, 768]" = torch.ops.aten.view.default(view_146, [4096, 768]);  view_146 = None
        mm: "f32[4096, 2]" = torch.ops.aten.mm.default(view_147, permute_48);  view_147 = permute_48 = None
        view_148: "f32[4, 1024, 2]" = torch.ops.aten.view.default(mm, [4, 1024, 2]);  mm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1705 in forward, code: sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        eq: "b8[4, 1024]" = torch.ops.aten.eq.Scalar(arg0_1, 0);  arg0_1 = None
        convert_element_type: "i32[4, 1024]" = torch.ops.prims.convert_element_type.default(eq, torch.int32);  eq = None
        argmax: "i64[4]" = torch.ops.aten.argmax.default(convert_element_type, -1);  convert_element_type = None
        sub_25: "i64[4]" = torch.ops.aten.sub.Tensor(argmax, 1);  argmax = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1706 in forward, code: sequence_lengths = sequence_lengths % input_ids.shape[-1]
        remainder: "i64[4]" = torch.ops.aten.remainder.Scalar(sub_25, 1024);  sub_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1715 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        iota_2: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "f32[4, 2]" = torch.ops.aten.index.Tensor(view_148, [iota_2, remainder]);  view_148 = iota_2 = remainder = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1735 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        view_149: "f32[4, 2]" = torch.ops.aten.view.default(index, [-1, 2])
        view_150: "i64[4]" = torch.ops.aten.view.default(arg150_1, [-1]);  arg150_1 = None
        amax: "f32[4, 1]" = torch.ops.aten.amax.default(view_149, [1], True)
        sub_26: "f32[4, 2]" = torch.ops.aten.sub.Tensor(view_149, amax);  view_149 = amax = None
        exp: "f32[4, 2]" = torch.ops.aten.exp.default(sub_26)
        sum_1: "f32[4, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[4, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_27: "f32[4, 2]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
        ne: "b8[4]" = torch.ops.aten.ne.Scalar(view_150, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[4]" = torch.ops.aten.where.self(ne, view_150, full_default_2);  ne = full_default_2 = None
        unsqueeze_27: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[4, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_27);  sub_27 = unsqueeze_27 = None
        squeeze: "f32[4]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1: "b8[4]" = torch.ops.aten.ne.Scalar(view_150, -100)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2: "b8[4]" = torch.ops.aten.ne.Scalar(view_150, -100);  view_150 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type_1);  sum_3 = convert_element_type_1 = None
        return (div, index, permute_1, permute_2, permute_5, permute_6, permute_9, permute_10, permute_13, permute_14, permute_17, permute_18, permute_21, permute_22, permute_25, permute_26, permute_29, permute_30, permute_33, permute_34, permute_37, permute_38, permute_41, permute_42, permute_45, permute_46)
        