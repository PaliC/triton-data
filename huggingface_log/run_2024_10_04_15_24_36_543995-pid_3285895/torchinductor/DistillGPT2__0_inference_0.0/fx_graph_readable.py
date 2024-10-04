class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "f32[50257, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[2304]", arg6_1: "f32[768, 2304]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[2304]", arg18_1: "f32[768, 2304]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[768]", arg26_1: "f32[3072, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[2304]", arg30_1: "f32[768, 2304]", arg31_1: "f32[768]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[3072, 768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[2304]", arg42_1: "f32[768, 2304]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[3072, 768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[2304]", arg54_1: "f32[768, 2304]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[2304]", arg66_1: "f32[768, 2304]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[3072, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "i64[16, 512]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1005 in forward, code: input_ids = input_ids.view(-1, input_shape[-1])
        view: "i64[16, 512]" = torch.ops.aten.view.default(arg0_1, [-1, 512]);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1024 in forward, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1025 in forward, code: position_ids = position_ids.unsqueeze(0)
        unsqueeze: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1028 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[16, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, view);  view = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1029 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 512, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = unsqueeze = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1030 in forward, code: hidden_states = inputs_embeds + position_embeds
        add: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:158 in _make_causal_mask, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        full_default: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:159 in _make_causal_mask, code: mask_cond = torch.arange(mask.size(-1), device=device)
        iota_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:160 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        add_1: "i64[512]" = torch.ops.aten.add.Tensor(iota_1, 1)
        view_1: "i64[512, 1]" = torch.ops.aten.view.default(add_1, [512, 1]);  add_1 = None
        lt: "b8[512, 512]" = torch.ops.aten.lt.Tensor(iota_1, view_1);  iota_1 = view_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[512, 512]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1]" = var_mean[0]
        getitem_1: "f32[16, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_3: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_2: "f32[8192, 768]" = torch.ops.aten.view.default(add_3, [-1, 768]);  add_3 = None
        addmm: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg5_1, view_2, arg6_1);  arg5_1 = view_2 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_3: "f32[16, 512, 2304]" = torch.ops.aten.view.default(addmm, [16, 512, 2304]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_3, 768, 2);  view_3 = None
        getitem_2: "f32[16, 512, 768]" = split[0]
        getitem_3: "f32[16, 512, 768]" = split[1]
        getitem_4: "f32[16, 512, 768]" = split[2];  split = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_4: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_2, [16, 512, 12, 64]);  getitem_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_5: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_3, [16, 512, 12, 64]);  getitem_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_6: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_4, [16, 512, 12, 64]);  getitem_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_2: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_3: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_4: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        expand_2: "f32[16, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_4, [16, 1, 512, 512]);  unsqueeze_4 = None
        expand_3: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(expand_2, [16, 12, 512, 512]);  expand_2 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute, permute_1, permute_2, expand_3, False);  permute = expand_3 = None
        getitem_5: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_7: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_3, [16, 512, 768]);  permute_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_8: "f32[8192, 768]" = torch.ops.aten.view.default(view_7, [-1, 768]);  view_7 = None
        addmm_1: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg7_1, view_8, arg8_1);  arg7_1 = view_8 = arg8_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_9: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_1, [16, 512, 768]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_4: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_9, add);  view_9 = add = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_9: "f32[16, 512, 1]" = var_mean_1[0]
        getitem_10: "f32[16, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_10);  getitem_10 = None
        mul_2: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_6: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_10: "f32[8192, 768]" = torch.ops.aten.view.default(add_6, [-1, 768]);  add_6 = None
        addmm_2: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg11_1, view_10, arg12_1);  arg11_1 = view_10 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_11: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_2, [16, 512, 3072]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_11, 0.5)
        pow_1: "f32[16, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_11, 3.0)
        mul_5: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(view_11, mul_5);  view_11 = mul_5 = None
        mul_6: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[16, 512, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_12: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg13_1, view_12, arg14_1);  arg13_1 = view_12 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_13: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_3, [16, 512, 768]);  addmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_9: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add_4, view_13);  add_4 = view_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_11: "f32[16, 512, 1]" = var_mean_2[0]
        getitem_12: "f32[16, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_12);  getitem_12 = None
        mul_8: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_11: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_14: "f32[8192, 768]" = torch.ops.aten.view.default(add_11, [-1, 768]);  add_11 = None
        addmm_4: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg17_1, view_14, arg18_1);  arg17_1 = view_14 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_15: "f32[16, 512, 2304]" = torch.ops.aten.view.default(addmm_4, [16, 512, 2304]);  addmm_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_15, 768, 2);  view_15 = None
        getitem_13: "f32[16, 512, 768]" = split_1[0]
        getitem_14: "f32[16, 512, 768]" = split_1[1]
        getitem_15: "f32[16, 512, 768]" = split_1[2];  split_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_16: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_13, [16, 512, 12, 64]);  getitem_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_4: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_17: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_14, [16, 512, 12, 64]);  getitem_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_5: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_18: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_15, [16, 512, 12, 64]);  getitem_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_6: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_5: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_6: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 1);  unsqueeze_5 = None
        expand_5: "f32[16, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_6, [16, 1, 512, 512]);  unsqueeze_6 = None
        expand_6: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(expand_5, [16, 12, 512, 512]);  expand_5 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_4, permute_5, permute_6, expand_6, False);  permute_4 = expand_6 = None
        getitem_16: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_19: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_7, [16, 512, 768]);  permute_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_20: "f32[8192, 768]" = torch.ops.aten.view.default(view_19, [-1, 768]);  view_19 = None
        addmm_5: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg19_1, view_20, arg20_1);  arg19_1 = view_20 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_21: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_5, [16, 512, 768]);  addmm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_12: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_21, add_9);  view_21 = add_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_20: "f32[16, 512, 1]" = var_mean_3[0]
        getitem_21: "f32[16, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_21);  getitem_21 = None
        mul_10: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_14: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_22: "f32[8192, 768]" = torch.ops.aten.view.default(add_14, [-1, 768]);  add_14 = None
        addmm_6: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg23_1, view_22, arg24_1);  arg23_1 = view_22 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_23: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_6, [16, 512, 3072]);  addmm_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
        pow_2: "f32[16, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
        mul_13: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(view_23, mul_13);  view_23 = mul_13 = None
        mul_14: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[16, 512, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_24: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg25_1, view_24, arg26_1);  arg25_1 = view_24 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_25: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_7, [16, 512, 768]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_17: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add_12, view_25);  add_12 = view_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22: "f32[16, 512, 1]" = var_mean_4[0]
        getitem_23: "f32[16, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  getitem_23 = None
        mul_16: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_19: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_26: "f32[8192, 768]" = torch.ops.aten.view.default(add_19, [-1, 768]);  add_19 = None
        addmm_8: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg29_1, view_26, arg30_1);  arg29_1 = view_26 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_27: "f32[16, 512, 2304]" = torch.ops.aten.view.default(addmm_8, [16, 512, 2304]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_27, 768, 2);  view_27 = None
        getitem_24: "f32[16, 512, 768]" = split_2[0]
        getitem_25: "f32[16, 512, 768]" = split_2[1]
        getitem_26: "f32[16, 512, 768]" = split_2[2];  split_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_28: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_24, [16, 512, 12, 64]);  getitem_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_8: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_29: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_25, [16, 512, 12, 64]);  getitem_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_9: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_30: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_26, [16, 512, 12, 64]);  getitem_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_10: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_7: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_8: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 1);  unsqueeze_7 = None
        expand_8: "f32[16, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_8, [16, 1, 512, 512]);  unsqueeze_8 = None
        expand_9: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(expand_8, [16, 12, 512, 512]);  expand_8 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_8, permute_9, permute_10, expand_9, False);  permute_8 = expand_9 = None
        getitem_27: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_31: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_11, [16, 512, 768]);  permute_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_32: "f32[8192, 768]" = torch.ops.aten.view.default(view_31, [-1, 768]);  view_31 = None
        addmm_9: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg31_1, view_32, arg32_1);  arg31_1 = view_32 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_33: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_9, [16, 512, 768]);  addmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_20: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_33, add_17);  view_33 = add_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_31: "f32[16, 512, 1]" = var_mean_5[0]
        getitem_32: "f32[16, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_32);  getitem_32 = None
        mul_18: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_19: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_22: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_34: "f32[8192, 768]" = torch.ops.aten.view.default(add_22, [-1, 768]);  add_22 = None
        addmm_10: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg35_1, view_34, arg36_1);  arg35_1 = view_34 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_35: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [16, 512, 3072]);  addmm_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
        pow_3: "f32[16, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_35, 3.0)
        mul_21: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(view_35, mul_21);  view_35 = mul_21 = None
        mul_22: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[16, 512, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_36: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg37_1, view_36, arg38_1);  arg37_1 = view_36 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_37: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_11, [16, 512, 768]);  addmm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_25: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add_20, view_37);  add_20 = view_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_33: "f32[16, 512, 1]" = var_mean_6[0]
        getitem_34: "f32[16, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_6: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_34);  getitem_34 = None
        mul_24: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_25: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_27: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_38: "f32[8192, 768]" = torch.ops.aten.view.default(add_27, [-1, 768]);  add_27 = None
        addmm_12: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg41_1, view_38, arg42_1);  arg41_1 = view_38 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_39: "f32[16, 512, 2304]" = torch.ops.aten.view.default(addmm_12, [16, 512, 2304]);  addmm_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_39, 768, 2);  view_39 = None
        getitem_35: "f32[16, 512, 768]" = split_3[0]
        getitem_36: "f32[16, 512, 768]" = split_3[1]
        getitem_37: "f32[16, 512, 768]" = split_3[2];  split_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_40: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_35, [16, 512, 12, 64]);  getitem_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_12: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_41: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_36, [16, 512, 12, 64]);  getitem_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_13: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_42: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_37, [16, 512, 12, 64]);  getitem_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_14: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_9: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_10: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        expand_11: "f32[16, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_10, [16, 1, 512, 512]);  unsqueeze_10 = None
        expand_12: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(expand_11, [16, 12, 512, 512]);  expand_11 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_12, permute_13, permute_14, expand_12, False);  permute_12 = expand_12 = None
        getitem_38: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_43: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_15, [16, 512, 768]);  permute_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_44: "f32[8192, 768]" = torch.ops.aten.view.default(view_43, [-1, 768]);  view_43 = None
        addmm_13: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg43_1, view_44, arg44_1);  arg43_1 = view_44 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_45: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_13, [16, 512, 768]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_28: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_45, add_25);  view_45 = add_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_42: "f32[16, 512, 1]" = var_mean_7[0]
        getitem_43: "f32[16, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_43);  getitem_43 = None
        mul_26: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_27: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_30: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_46: "f32[8192, 768]" = torch.ops.aten.view.default(add_30, [-1, 768]);  add_30 = None
        addmm_14: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg47_1, view_46, arg48_1);  arg47_1 = view_46 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_47: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_14, [16, 512, 3072]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_47, 0.5)
        pow_4: "f32[16, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_47, 3.0)
        mul_29: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(view_47, mul_29);  view_47 = mul_29 = None
        mul_30: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f32[16, 512, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_48: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg49_1, view_48, arg50_1);  arg49_1 = view_48 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_49: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_15, [16, 512, 768]);  addmm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_33: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add_28, view_49);  add_28 = view_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_44: "f32[16, 512, 1]" = var_mean_8[0]
        getitem_45: "f32[16, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_8: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_45);  getitem_45 = None
        mul_32: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_33: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_35: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_50: "f32[8192, 768]" = torch.ops.aten.view.default(add_35, [-1, 768]);  add_35 = None
        addmm_16: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg53_1, view_50, arg54_1);  arg53_1 = view_50 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_51: "f32[16, 512, 2304]" = torch.ops.aten.view.default(addmm_16, [16, 512, 2304]);  addmm_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_51, 768, 2);  view_51 = None
        getitem_46: "f32[16, 512, 768]" = split_4[0]
        getitem_47: "f32[16, 512, 768]" = split_4[1]
        getitem_48: "f32[16, 512, 768]" = split_4[2];  split_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_52: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_46, [16, 512, 12, 64]);  getitem_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_16: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_53: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_47, [16, 512, 12, 64]);  getitem_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_17: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_54: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_48, [16, 512, 12, 64]);  getitem_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_18: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_11: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0)
        unsqueeze_12: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 1);  unsqueeze_11 = None
        expand_14: "f32[16, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_12, [16, 1, 512, 512]);  unsqueeze_12 = None
        expand_15: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(expand_14, [16, 12, 512, 512]);  expand_14 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_16, permute_17, permute_18, expand_15, False);  permute_16 = expand_15 = None
        getitem_49: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_55: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_19, [16, 512, 768]);  permute_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_56: "f32[8192, 768]" = torch.ops.aten.view.default(view_55, [-1, 768]);  view_55 = None
        addmm_17: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg55_1, view_56, arg56_1);  arg55_1 = view_56 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_57: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_17, [16, 512, 768]);  addmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_36: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_57, add_33);  view_57 = add_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_53: "f32[16, 512, 1]" = var_mean_9[0]
        getitem_54: "f32[16, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_37: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_54);  getitem_54 = None
        mul_34: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_35: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_38: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_58: "f32[8192, 768]" = torch.ops.aten.view.default(add_38, [-1, 768]);  add_38 = None
        addmm_18: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg59_1, view_58, arg60_1);  arg59_1 = view_58 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_59: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_18, [16, 512, 3072]);  addmm_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_59, 0.5)
        pow_5: "f32[16, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_59, 3.0)
        mul_37: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(view_59, mul_37);  view_59 = mul_37 = None
        mul_38: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[16, 512, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_60: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg61_1, view_60, arg62_1);  arg61_1 = view_60 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_61: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_19, [16, 512, 768]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_41: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add_36, view_61);  add_36 = view_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:614 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_55: "f32[16, 512, 1]" = var_mean_10[0]
        getitem_56: "f32[16, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_10: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_56);  getitem_56 = None
        mul_40: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_41: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_43: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_62: "f32[8192, 768]" = torch.ops.aten.view.default(add_43, [-1, 768]);  add_43 = None
        addmm_20: "f32[8192, 2304]" = torch.ops.aten.addmm.default(arg65_1, view_62, arg66_1);  arg65_1 = view_62 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_63: "f32[16, 512, 2304]" = torch.ops.aten.view.default(addmm_20, [16, 512, 2304]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:518 in forward, code: query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_63, 768, 2);  view_63 = None
        getitem_57: "f32[16, 512, 768]" = split_5[0]
        getitem_58: "f32[16, 512, 768]" = split_5[1]
        getitem_59: "f32[16, 512, 768]" = split_5[2];  split_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_64: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_57, [16, 512, 12, 64]);  getitem_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_20: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_65: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_58, [16, 512, 12, 64]);  getitem_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_21: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:280 in _split_heads, code: tensor = tensor.view(new_shape)
        view_66: "f32[16, 512, 12, 64]" = torch.ops.aten.view.default(getitem_59, [16, 512, 12, 64]);  getitem_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:281 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_22: "f32[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:545 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        unsqueeze_13: "f32[1, 512, 512]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_14: "f32[1, 1, 512, 512]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None
        expand_17: "f32[16, 1, 512, 512]" = torch.ops.aten.expand.default(unsqueeze_14, [16, 1, 512, 512]);  unsqueeze_14 = None
        expand_18: "f32[16, 12, 512, 512]" = torch.ops.aten.expand.default(expand_17, [16, 12, 512, 512]);  expand_17 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_20, permute_21, permute_22, expand_18, False);  permute_20 = expand_18 = None
        getitem_60: "f32[16, 12, 512, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:555 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[16, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:556 in forward, code: attn_output = attn_output.view(bsz, q_len, self.embed_dim)
        view_67: "f32[16, 512, 768]" = torch.ops.aten.view.default(permute_23, [16, 512, 768]);  permute_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_68: "f32[8192, 768]" = torch.ops.aten.view.default(view_67, [-1, 768]);  view_67 = None
        addmm_21: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg67_1, view_68, arg68_1);  arg67_1 = view_68 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_69: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_21, [16, 512, 768]);  addmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:626 in forward, code: hidden_states = attn_output + residual
        add_44: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(view_69, add_41);  view_69 = add_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:651 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_64: "f32[16, 512, 1]" = var_mean_11[0]
        getitem_65: "f32[16, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_45: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_65);  getitem_65 = None
        mul_42: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_43: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_46: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_70: "f32[8192, 768]" = torch.ops.aten.view.default(add_46, [-1, 768]);  add_46 = None
        addmm_22: "f32[8192, 3072]" = torch.ops.aten.addmm.default(arg71_1, view_70, arg72_1);  arg71_1 = view_70 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_71: "f32[16, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [16, 512, 3072]);  addmm_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:56 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(view_71, 0.5)
        pow_6: "f32[16, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_71, 3.0)
        mul_45: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(view_71, mul_45);  view_71 = mul_45 = None
        mul_46: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f32[16, 512, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48: "f32[16, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[16, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:111 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_72: "f32[8192, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23: "f32[8192, 768]" = torch.ops.aten.addmm.default(arg73_1, view_72, arg74_1);  arg73_1 = view_72 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/pytorch_utils.py:112 in forward, code: x = x.view(size_out)
        view_73: "f32[16, 512, 768]" = torch.ops.aten.view.default(addmm_23, [16, 512, 768]);  addmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:654 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_49: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(add_44, view_73);  add_44 = view_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1156 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_66: "f32[16, 512, 1]" = var_mean_12[0]
        getitem_67: "f32[16, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[16, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[16, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12: "f32[16, 512, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_67);  add_49 = getitem_67 = None
        mul_48: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49: "f32[16, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_51: "f32[16, 512, 768]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1158 in forward, code: hidden_states = hidden_states.view(output_shape)
        view_74: "f32[16, 512, 768]" = torch.ops.aten.view.default(add_51, [-1, 512, 768]);  add_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1338 in forward, code: lm_logits = self.lm_head(hidden_states)
        permute_24: "f32[768, 50257]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_75: "f32[8192, 768]" = torch.ops.aten.view.default(view_74, [8192, 768]);  view_74 = None
        
        # No stacktrace found for following nodes
        full_default_4: "f32[768, 3]" = torch.ops.aten.full.default([768, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[768, 50260]" = torch.ops.aten.cat.default([permute_24, full_default_4], 1);  permute_24 = full_default_4 = None
        mm_default: "f32[8192, 50260]" = torch.ops.aten.mm.default(view_75, cat_default);  view_75 = cat_default = None
        slice_tensor: "f32[8192, 50257]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1338 in forward, code: lm_logits = self.lm_head(hidden_states)
        view_76: "f32[16, 512, 50257]" = torch.ops.aten.view.default(slice_tensor, [16, 512, 50257]);  slice_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1345 in forward, code: shift_logits = lm_logits[..., :-1, :].contiguous()
        slice_15: "f32[16, 511, 50257]" = torch.ops.aten.slice.Tensor(view_76, 1, 0, -1)
        clone_13: "f32[16, 511, 50257]" = torch.ops.aten.clone.default(slice_15, memory_format = torch.contiguous_format);  slice_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1346 in forward, code: shift_labels = labels[..., 1:].contiguous()
        slice_17: "i64[16, 511]" = torch.ops.aten.slice.Tensor(arg77_1, 1, 1, 9223372036854775807);  arg77_1 = None
        clone_14: "i64[16, 511]" = torch.ops.aten.clone.default(slice_17, memory_format = torch.contiguous_format);  slice_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:1349 in forward, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        view_77: "f32[8176, 50257]" = torch.ops.aten.view.default(clone_13, [-1, 50257]);  clone_13 = None
        view_78: "i64[8176]" = torch.ops.aten.view.default(clone_14, [-1]);  clone_14 = None
        amax: "f32[8176, 1]" = torch.ops.aten.amax.default(view_77, [1], True)
        sub_13: "f32[8176, 50257]" = torch.ops.aten.sub.Tensor(view_77, amax);  view_77 = amax = None
        exp: "f32[8176, 50257]" = torch.ops.aten.exp.default(sub_13)
        sum_1: "f32[8176, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8176, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_14: "f32[8176, 50257]" = torch.ops.aten.sub.Tensor(sub_13, log);  sub_13 = log = None
        ne: "b8[8176]" = torch.ops.aten.ne.Scalar(view_78, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[8176]" = torch.ops.aten.where.self(ne, view_78, full_default_2);  ne = full_default_2 = None
        unsqueeze_15: "i64[8176, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[8176, 1]" = torch.ops.aten.gather.default(sub_14, 1, unsqueeze_15);  sub_14 = unsqueeze_15 = None
        squeeze: "f32[8176]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8176]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1: "b8[8176]" = torch.ops.aten.ne.Scalar(view_78, -100)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8176]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        ne_2: "b8[8176]" = torch.ops.aten.ne.Scalar(view_78, -100);  view_78 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None
        return (div, view_76, permute_1, permute_2, permute_5, permute_6, permute_9, permute_10, permute_13, permute_14, permute_17, permute_18, permute_21, permute_22)
        