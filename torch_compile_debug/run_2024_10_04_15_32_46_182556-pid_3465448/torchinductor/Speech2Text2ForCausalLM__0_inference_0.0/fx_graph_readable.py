class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[256, 128]", arg1_1: "f32[10000, 256]", arg2_1: "f32[1026, 256]", arg3_1: "f32[256, 256]", arg4_1: "f32[256]", arg5_1: "f32[256, 256]", arg6_1: "f32[256]", arg7_1: "f32[256, 256]", arg8_1: "f32[256]", arg9_1: "f32[256, 256]", arg10_1: "f32[256]", arg11_1: "f32[256]", arg12_1: "f32[256]", arg13_1: "f32[2048, 256]", arg14_1: "f32[2048]", arg15_1: "f32[256, 2048]", arg16_1: "f32[256]", arg17_1: "f32[256]", arg18_1: "f32[256]", arg19_1: "f32[256, 256]", arg20_1: "f32[256]", arg21_1: "f32[256, 256]", arg22_1: "f32[256]", arg23_1: "f32[256, 256]", arg24_1: "f32[256]", arg25_1: "f32[256, 256]", arg26_1: "f32[256]", arg27_1: "f32[256]", arg28_1: "f32[256]", arg29_1: "f32[2048, 256]", arg30_1: "f32[2048]", arg31_1: "f32[256, 2048]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[256]", arg35_1: "f32[256, 256]", arg36_1: "f32[256]", arg37_1: "f32[256, 256]", arg38_1: "f32[256]", arg39_1: "f32[256, 256]", arg40_1: "f32[256]", arg41_1: "f32[256, 256]", arg42_1: "f32[256]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[2048, 256]", arg46_1: "f32[2048]", arg47_1: "f32[256, 2048]", arg48_1: "f32[256]", arg49_1: "f32[256]", arg50_1: "f32[256]", arg51_1: "f32[256, 256]", arg52_1: "f32[256]", arg53_1: "f32[256, 256]", arg54_1: "f32[256]", arg55_1: "f32[256, 256]", arg56_1: "f32[256]", arg57_1: "f32[256, 256]", arg58_1: "f32[256]", arg59_1: "f32[256]", arg60_1: "f32[256]", arg61_1: "f32[2048, 256]", arg62_1: "f32[2048]", arg63_1: "f32[256, 2048]", arg64_1: "f32[256]", arg65_1: "f32[256]", arg66_1: "f32[256]", arg67_1: "f32[256, 256]", arg68_1: "f32[256]", arg69_1: "f32[256, 256]", arg70_1: "f32[256]", arg71_1: "f32[256, 256]", arg72_1: "f32[256]", arg73_1: "f32[256, 256]", arg74_1: "f32[256]", arg75_1: "f32[256]", arg76_1: "f32[256]", arg77_1: "f32[2048, 256]", arg78_1: "f32[2048]", arg79_1: "f32[256, 2048]", arg80_1: "f32[256]", arg81_1: "f32[256]", arg82_1: "f32[256]", arg83_1: "f32[256, 256]", arg84_1: "f32[256]", arg85_1: "f32[256, 256]", arg86_1: "f32[256]", arg87_1: "f32[256, 256]", arg88_1: "f32[256]", arg89_1: "f32[256, 256]", arg90_1: "f32[256]", arg91_1: "f32[256]", arg92_1: "f32[256]", arg93_1: "f32[2048, 256]", arg94_1: "f32[2048]", arg95_1: "f32[256, 2048]", arg96_1: "f32[256]", arg97_1: "f32[256]", arg98_1: "f32[256]", arg99_1: "i64[256, 128]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:550 in forward, code: input_ids = input_ids.view(-1, input_shape[-1])
        view: "i64[256, 128]" = torch.ops.aten.view.default(arg0_1, [-1, 128]);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:560 in forward, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embedding: "f32[256, 128, 256]" = torch.ops.aten.embedding.default(arg1_1, view, 1)
        mul: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(embedding, 16.0);  embedding = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:158 in _make_causal_mask, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        full_default: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:159 in _make_causal_mask, code: mask_cond = torch.arange(mask.size(-1), device=device)
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:160 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 1)
        view_1: "i64[128, 1]" = torch.ops.aten.view.default(add, [128, 1]);  add = None
        lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 128]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:104 in create_position_ids_from_input_ids, code: mask = input_ids.ne(padding_idx).int()
        ne: "b8[256, 128]" = torch.ops.aten.ne.Scalar(view, 1);  view = None
        convert_element_type: "i32[256, 128]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:105 in create_position_ids_from_input_ids, code: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        cumsum: "i64[256, 128]" = torch.ops.aten.cumsum.default(convert_element_type, 1)
        convert_element_type_1: "i32[256, 128]" = torch.ops.prims.convert_element_type.default(cumsum, torch.int32);  cumsum = None
        add_1: "i32[256, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
        mul_1: "i32[256, 128]" = torch.ops.aten.mul.Tensor(add_1, convert_element_type);  add_1 = convert_element_type = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:106 in create_position_ids_from_input_ids, code: return incremental_indices.long() + padding_idx
        convert_element_type_2: "i64[256, 128]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        add_2: "i64[256, 128]" = torch.ops.aten.add.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:90 in forward, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
        view_2: "i64[32768]" = torch.ops.aten.view.default(add_2, [-1]);  add_2 = None
        index: "f32[32768, 256]" = torch.ops.aten.index.Tensor(arg2_1, [view_2]);  arg2_1 = view_2 = None
        view_3: "f32[256, 128, 256]" = torch.ops.aten.view.default(index, [256, 128, -1]);  index = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:576 in forward, code: hidden_states = inputs_embeds + positions
        add_3: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul, view_3);  mul = view_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:164 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_4: "f32[32768, 256]" = torch.ops.aten.view.default(add_3, [32768, 256])
        permute: "f32[256, 256]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg4_1, view_4, permute);  arg4_1 = view_4 = permute = None
        view_5: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm, [256, 128, 256]);  addmm = None
        mul_2: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(view_5, 0.125);  view_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:189 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_6: "f32[32768, 256]" = torch.ops.aten.view.default(add_3, [32768, 256])
        permute_1: "f32[256, 256]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm_1: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg6_1, view_6, permute_1);  arg6_1 = view_6 = permute_1 = None
        view_7: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_1, [256, 128, 256]);  addmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_8: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_7, [256, -1, 4, 64]);  view_7 = None
        permute_2: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_1: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:190 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_9: "f32[32768, 256]" = torch.ops.aten.view.default(add_3, [32768, 256])
        permute_3: "f32[256, 256]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_2: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg8_1, view_9, permute_3);  arg8_1 = view_9 = permute_3 = None
        view_10: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_2, [256, 128, 256]);  addmm_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_11: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_10, [256, -1, 4, 64]);  view_10 = None
        permute_4: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        clone_2: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_12: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(mul_2, [256, 128, 4, 64]);  mul_2 = None
        permute_5: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_3: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:203 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_13: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_3, [1024, -1, 64]);  clone_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:204 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_14: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_1, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:205 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_15: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_2, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:208 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_6: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        bmm: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_13, permute_6);  view_13 = permute_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:221 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_16: "f32[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, [256, 4, 128, 128]);  bmm = None
        unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_3: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_1: "f32[256, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_3, [256, 1, 128, 128]);  unsqueeze_3 = None
        add_4: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_16, expand_1);  view_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:222 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_17: "f32[1024, 128, 128]" = torch.ops.aten.view.default(add_4, [1024, 128, 128]);  add_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:224 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax: "f32[1024, 128, 1]" = torch.ops.aten.amax.default(view_17, [-1], True)
        sub: "f32[1024, 128, 128]" = torch.ops.aten.sub.Tensor(view_17, amax);  view_17 = amax = None
        exp: "f32[1024, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1024, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1024, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:247 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_1: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(div, view_15);  div = view_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:255 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_18: "f32[256, 4, 128, 64]" = torch.ops.aten.view.default(bmm_1, [256, 4, 128, 64]);  bmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:256 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_7: "f32[256, 128, 4, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:260 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_5: "f32[256, 128, 4, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_19: "f32[256, 128, 256]" = torch.ops.aten.view.default(clone_5, [256, 128, 256]);  clone_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:262 in forward, code: attn_output = self.out_proj(attn_output)
        view_20: "f32[32768, 256]" = torch.ops.aten.view.default(view_19, [32768, 256]);  view_19 = None
        permute_8: "f32[256, 256]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_3: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg10_1, view_20, permute_8);  arg10_1 = view_20 = permute_8 = None
        view_21: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_3, [256, 128, 256]);  addmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:341 in forward, code: hidden_states = residual + hidden_states
        add_5: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_3, view_21);  add_3 = view_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:342 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean[0]
        getitem_1: "f32[256, 128, 1]" = var_mean[1];  var_mean = None
        add_6: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_5, getitem_1);  add_5 = getitem_1 = None
        mul_3: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_4: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
        add_7: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_4, arg12_1);  mul_4 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:369 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_22: "f32[32768, 256]" = torch.ops.aten.view.default(add_7, [32768, 256])
        permute_9: "f32[256, 2048]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_4: "f32[32768, 2048]" = torch.ops.aten.addmm.default(arg14_1, view_22, permute_9);  arg14_1 = view_22 = permute_9 = None
        view_23: "f32[256, 128, 2048]" = torch.ops.aten.view.default(addmm_4, [256, 128, 2048]);  addmm_4 = None
        relu: "f32[256, 128, 2048]" = torch.ops.aten.relu.default(view_23);  view_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:371 in forward, code: hidden_states = self.fc2(hidden_states)
        view_24: "f32[32768, 2048]" = torch.ops.aten.view.default(relu, [32768, 2048]);  relu = None
        permute_10: "f32[2048, 256]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_5: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg16_1, view_24, permute_10);  arg16_1 = view_24 = permute_10 = None
        view_25: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_5, [256, 128, 256]);  addmm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:373 in forward, code: hidden_states = residual + hidden_states
        add_8: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_7, view_25);  add_7 = view_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:374 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_2: "f32[256, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[256, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_9: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_8, getitem_3);  add_8 = getitem_3 = None
        mul_5: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_5, arg17_1);  mul_5 = arg17_1 = None
        add_10: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_6, arg18_1);  mul_6 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:164 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_26: "f32[32768, 256]" = torch.ops.aten.view.default(add_10, [32768, 256])
        permute_11: "f32[256, 256]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_6: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg20_1, view_26, permute_11);  arg20_1 = view_26 = permute_11 = None
        view_27: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_6, [256, 128, 256]);  addmm_6 = None
        mul_7: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(view_27, 0.125);  view_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:189 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_28: "f32[32768, 256]" = torch.ops.aten.view.default(add_10, [32768, 256])
        permute_12: "f32[256, 256]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_7: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg22_1, view_28, permute_12);  arg22_1 = view_28 = permute_12 = None
        view_29: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_7, [256, 128, 256]);  addmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_30: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_29, [256, -1, 4, 64]);  view_29 = None
        permute_13: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        clone_9: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:190 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_31: "f32[32768, 256]" = torch.ops.aten.view.default(add_10, [32768, 256])
        permute_14: "f32[256, 256]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_8: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg24_1, view_31, permute_14);  arg24_1 = view_31 = permute_14 = None
        view_32: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_8, [256, 128, 256]);  addmm_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_33: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_32, [256, -1, 4, 64]);  view_32 = None
        permute_15: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_10: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_34: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(mul_7, [256, 128, 4, 64]);  mul_7 = None
        permute_16: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        clone_11: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:203 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_35: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_11, [1024, -1, 64]);  clone_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:204 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_36: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_9, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:205 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_37: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_10, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:208 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_17: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        bmm_2: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_35, permute_17);  view_35 = permute_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:221 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_38: "f32[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_2, [256, 4, 128, 128]);  bmm_2 = None
        add_11: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_38, expand_1);  view_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:222 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_39: "f32[1024, 128, 128]" = torch.ops.aten.view.default(add_11, [1024, 128, 128]);  add_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:224 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_1: "f32[1024, 128, 1]" = torch.ops.aten.amax.default(view_39, [-1], True)
        sub_3: "f32[1024, 128, 128]" = torch.ops.aten.sub.Tensor(view_39, amax_1);  view_39 = amax_1 = None
        exp_1: "f32[1024, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_2: "f32[1024, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[1024, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:247 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_3: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(div_1, view_37);  div_1 = view_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:255 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_40: "f32[256, 4, 128, 64]" = torch.ops.aten.view.default(bmm_3, [256, 4, 128, 64]);  bmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:256 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_18: "f32[256, 128, 4, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:260 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_13: "f32[256, 128, 4, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_41: "f32[256, 128, 256]" = torch.ops.aten.view.default(clone_13, [256, 128, 256]);  clone_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:262 in forward, code: attn_output = self.out_proj(attn_output)
        view_42: "f32[32768, 256]" = torch.ops.aten.view.default(view_41, [32768, 256]);  view_41 = None
        permute_19: "f32[256, 256]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_9: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg26_1, view_42, permute_19);  arg26_1 = view_42 = permute_19 = None
        view_43: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_9, [256, 128, 256]);  addmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:341 in forward, code: hidden_states = residual + hidden_states
        add_12: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_10, view_43);  add_10 = view_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:342 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_4: "f32[256, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[256, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_13: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_4: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_12, getitem_5);  add_12 = getitem_5 = None
        mul_8: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_8, arg27_1);  mul_8 = arg27_1 = None
        add_14: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_9, arg28_1);  mul_9 = arg28_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:369 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_44: "f32[32768, 256]" = torch.ops.aten.view.default(add_14, [32768, 256])
        permute_20: "f32[256, 2048]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_10: "f32[32768, 2048]" = torch.ops.aten.addmm.default(arg30_1, view_44, permute_20);  arg30_1 = view_44 = permute_20 = None
        view_45: "f32[256, 128, 2048]" = torch.ops.aten.view.default(addmm_10, [256, 128, 2048]);  addmm_10 = None
        relu_1: "f32[256, 128, 2048]" = torch.ops.aten.relu.default(view_45);  view_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:371 in forward, code: hidden_states = self.fc2(hidden_states)
        view_46: "f32[32768, 2048]" = torch.ops.aten.view.default(relu_1, [32768, 2048]);  relu_1 = None
        permute_21: "f32[2048, 256]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_11: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg32_1, view_46, permute_21);  arg32_1 = view_46 = permute_21 = None
        view_47: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_11, [256, 128, 256]);  addmm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:373 in forward, code: hidden_states = residual + hidden_states
        add_15: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_14, view_47);  add_14 = view_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:374 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_6: "f32[256, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[256, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_15, getitem_7);  add_15 = getitem_7 = None
        mul_10: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_10, arg33_1);  mul_10 = arg33_1 = None
        add_17: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_11, arg34_1);  mul_11 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:164 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_48: "f32[32768, 256]" = torch.ops.aten.view.default(add_17, [32768, 256])
        permute_22: "f32[256, 256]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_12: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg36_1, view_48, permute_22);  arg36_1 = view_48 = permute_22 = None
        view_49: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_12, [256, 128, 256]);  addmm_12 = None
        mul_12: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(view_49, 0.125);  view_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:189 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_50: "f32[32768, 256]" = torch.ops.aten.view.default(add_17, [32768, 256])
        permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_13: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg38_1, view_50, permute_23);  arg38_1 = view_50 = permute_23 = None
        view_51: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_13, [256, 128, 256]);  addmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_52: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_51, [256, -1, 4, 64]);  view_51 = None
        permute_24: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_17: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:190 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_53: "f32[32768, 256]" = torch.ops.aten.view.default(add_17, [32768, 256])
        permute_25: "f32[256, 256]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_14: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg40_1, view_53, permute_25);  arg40_1 = view_53 = permute_25 = None
        view_54: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_14, [256, 128, 256]);  addmm_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_55: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_54, [256, -1, 4, 64]);  view_54 = None
        permute_26: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        clone_18: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_56: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(mul_12, [256, 128, 4, 64]);  mul_12 = None
        permute_27: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        clone_19: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:203 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_57: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_19, [1024, -1, 64]);  clone_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:204 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_58: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_17, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:205 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_59: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_18, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:208 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_28: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        bmm_4: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_57, permute_28);  view_57 = permute_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:221 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_60: "f32[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_4, [256, 4, 128, 128]);  bmm_4 = None
        add_18: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_60, expand_1);  view_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:222 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_61: "f32[1024, 128, 128]" = torch.ops.aten.view.default(add_18, [1024, 128, 128]);  add_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:224 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_2: "f32[1024, 128, 1]" = torch.ops.aten.amax.default(view_61, [-1], True)
        sub_6: "f32[1024, 128, 128]" = torch.ops.aten.sub.Tensor(view_61, amax_2);  view_61 = amax_2 = None
        exp_2: "f32[1024, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_3: "f32[1024, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[1024, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:247 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_5: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(div_2, view_59);  div_2 = view_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:255 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_62: "f32[256, 4, 128, 64]" = torch.ops.aten.view.default(bmm_5, [256, 4, 128, 64]);  bmm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:256 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_29: "f32[256, 128, 4, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:260 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_21: "f32[256, 128, 4, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_63: "f32[256, 128, 256]" = torch.ops.aten.view.default(clone_21, [256, 128, 256]);  clone_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:262 in forward, code: attn_output = self.out_proj(attn_output)
        view_64: "f32[32768, 256]" = torch.ops.aten.view.default(view_63, [32768, 256]);  view_63 = None
        permute_30: "f32[256, 256]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_15: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg42_1, view_64, permute_30);  arg42_1 = view_64 = permute_30 = None
        view_65: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_15, [256, 128, 256]);  addmm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:341 in forward, code: hidden_states = residual + hidden_states
        add_19: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_17, view_65);  add_17 = view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:342 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_8: "f32[256, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[256, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_20: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_7: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_19, getitem_9);  add_19 = getitem_9 = None
        mul_13: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = rsqrt_4 = None
        mul_14: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_13, arg43_1);  mul_13 = arg43_1 = None
        add_21: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_14, arg44_1);  mul_14 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:369 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_66: "f32[32768, 256]" = torch.ops.aten.view.default(add_21, [32768, 256])
        permute_31: "f32[256, 2048]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_16: "f32[32768, 2048]" = torch.ops.aten.addmm.default(arg46_1, view_66, permute_31);  arg46_1 = view_66 = permute_31 = None
        view_67: "f32[256, 128, 2048]" = torch.ops.aten.view.default(addmm_16, [256, 128, 2048]);  addmm_16 = None
        relu_2: "f32[256, 128, 2048]" = torch.ops.aten.relu.default(view_67);  view_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:371 in forward, code: hidden_states = self.fc2(hidden_states)
        view_68: "f32[32768, 2048]" = torch.ops.aten.view.default(relu_2, [32768, 2048]);  relu_2 = None
        permute_32: "f32[2048, 256]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_17: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg48_1, view_68, permute_32);  arg48_1 = view_68 = permute_32 = None
        view_69: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_17, [256, 128, 256]);  addmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:373 in forward, code: hidden_states = residual + hidden_states
        add_22: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_21, view_69);  add_21 = view_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:374 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_10: "f32[256, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[256, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_8: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_22, getitem_11);  add_22 = getitem_11 = None
        mul_15: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_16: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_15, arg49_1);  mul_15 = arg49_1 = None
        add_24: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_16, arg50_1);  mul_16 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:164 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_70: "f32[32768, 256]" = torch.ops.aten.view.default(add_24, [32768, 256])
        permute_33: "f32[256, 256]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_18: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg52_1, view_70, permute_33);  arg52_1 = view_70 = permute_33 = None
        view_71: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_18, [256, 128, 256]);  addmm_18 = None
        mul_17: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(view_71, 0.125);  view_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:189 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_72: "f32[32768, 256]" = torch.ops.aten.view.default(add_24, [32768, 256])
        permute_34: "f32[256, 256]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_19: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg54_1, view_72, permute_34);  arg54_1 = view_72 = permute_34 = None
        view_73: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_19, [256, 128, 256]);  addmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_74: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_73, [256, -1, 4, 64]);  view_73 = None
        permute_35: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        clone_25: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:190 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_75: "f32[32768, 256]" = torch.ops.aten.view.default(add_24, [32768, 256])
        permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_20: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg56_1, view_75, permute_36);  arg56_1 = view_75 = permute_36 = None
        view_76: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_20, [256, 128, 256]);  addmm_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_77: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_76, [256, -1, 4, 64]);  view_76 = None
        permute_37: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        clone_26: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_78: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(mul_17, [256, 128, 4, 64]);  mul_17 = None
        permute_38: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        clone_27: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:203 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_79: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_27, [1024, -1, 64]);  clone_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:204 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_80: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_25, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:205 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_81: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_26, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:208 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_39: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
        bmm_6: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_79, permute_39);  view_79 = permute_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:221 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_82: "f32[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_6, [256, 4, 128, 128]);  bmm_6 = None
        add_25: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_82, expand_1);  view_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:222 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_83: "f32[1024, 128, 128]" = torch.ops.aten.view.default(add_25, [1024, 128, 128]);  add_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:224 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_3: "f32[1024, 128, 1]" = torch.ops.aten.amax.default(view_83, [-1], True)
        sub_9: "f32[1024, 128, 128]" = torch.ops.aten.sub.Tensor(view_83, amax_3);  view_83 = amax_3 = None
        exp_3: "f32[1024, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_4: "f32[1024, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[1024, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:247 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_7: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(div_3, view_81);  div_3 = view_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:255 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_84: "f32[256, 4, 128, 64]" = torch.ops.aten.view.default(bmm_7, [256, 4, 128, 64]);  bmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:256 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_40: "f32[256, 128, 4, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:260 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_29: "f32[256, 128, 4, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_85: "f32[256, 128, 256]" = torch.ops.aten.view.default(clone_29, [256, 128, 256]);  clone_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:262 in forward, code: attn_output = self.out_proj(attn_output)
        view_86: "f32[32768, 256]" = torch.ops.aten.view.default(view_85, [32768, 256]);  view_85 = None
        permute_41: "f32[256, 256]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_21: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg58_1, view_86, permute_41);  arg58_1 = view_86 = permute_41 = None
        view_87: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_21, [256, 128, 256]);  addmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:341 in forward, code: hidden_states = residual + hidden_states
        add_26: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_24, view_87);  add_24 = view_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:342 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_12: "f32[256, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[256, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_27: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_10: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_26, getitem_13);  add_26 = getitem_13 = None
        mul_18: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = rsqrt_6 = None
        mul_19: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_18, arg59_1);  mul_18 = arg59_1 = None
        add_28: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_19, arg60_1);  mul_19 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:369 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_88: "f32[32768, 256]" = torch.ops.aten.view.default(add_28, [32768, 256])
        permute_42: "f32[256, 2048]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_22: "f32[32768, 2048]" = torch.ops.aten.addmm.default(arg62_1, view_88, permute_42);  arg62_1 = view_88 = permute_42 = None
        view_89: "f32[256, 128, 2048]" = torch.ops.aten.view.default(addmm_22, [256, 128, 2048]);  addmm_22 = None
        relu_3: "f32[256, 128, 2048]" = torch.ops.aten.relu.default(view_89);  view_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:371 in forward, code: hidden_states = self.fc2(hidden_states)
        view_90: "f32[32768, 2048]" = torch.ops.aten.view.default(relu_3, [32768, 2048]);  relu_3 = None
        permute_43: "f32[2048, 256]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_23: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg64_1, view_90, permute_43);  arg64_1 = view_90 = permute_43 = None
        view_91: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_23, [256, 128, 256]);  addmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:373 in forward, code: hidden_states = residual + hidden_states
        add_29: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_28, view_91);  add_28 = view_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:374 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14: "f32[256, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[256, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_30: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_11: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  add_29 = getitem_15 = None
        mul_20: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_21: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_20, arg65_1);  mul_20 = arg65_1 = None
        add_31: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_21, arg66_1);  mul_21 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:164 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_92: "f32[32768, 256]" = torch.ops.aten.view.default(add_31, [32768, 256])
        permute_44: "f32[256, 256]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_24: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg68_1, view_92, permute_44);  arg68_1 = view_92 = permute_44 = None
        view_93: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_24, [256, 128, 256]);  addmm_24 = None
        mul_22: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(view_93, 0.125);  view_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:189 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_94: "f32[32768, 256]" = torch.ops.aten.view.default(add_31, [32768, 256])
        permute_45: "f32[256, 256]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_25: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg70_1, view_94, permute_45);  arg70_1 = view_94 = permute_45 = None
        view_95: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_25, [256, 128, 256]);  addmm_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_96: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_95, [256, -1, 4, 64]);  view_95 = None
        permute_46: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        clone_33: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:190 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_97: "f32[32768, 256]" = torch.ops.aten.view.default(add_31, [32768, 256])
        permute_47: "f32[256, 256]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_26: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg72_1, view_97, permute_47);  arg72_1 = view_97 = permute_47 = None
        view_98: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_26, [256, 128, 256]);  addmm_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_99: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_98, [256, -1, 4, 64]);  view_98 = None
        permute_48: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        clone_34: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_100: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(mul_22, [256, 128, 4, 64]);  mul_22 = None
        permute_49: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        clone_35: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:203 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_101: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_35, [1024, -1, 64]);  clone_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:204 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_102: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_33, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:205 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_103: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_34, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:208 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_50: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        bmm_8: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_101, permute_50);  view_101 = permute_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:221 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_104: "f32[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_8, [256, 4, 128, 128]);  bmm_8 = None
        add_32: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_104, expand_1);  view_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:222 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_105: "f32[1024, 128, 128]" = torch.ops.aten.view.default(add_32, [1024, 128, 128]);  add_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:224 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_4: "f32[1024, 128, 1]" = torch.ops.aten.amax.default(view_105, [-1], True)
        sub_12: "f32[1024, 128, 128]" = torch.ops.aten.sub.Tensor(view_105, amax_4);  view_105 = amax_4 = None
        exp_4: "f32[1024, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_5: "f32[1024, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[1024, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:247 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_9: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(div_4, view_103);  div_4 = view_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:255 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_106: "f32[256, 4, 128, 64]" = torch.ops.aten.view.default(bmm_9, [256, 4, 128, 64]);  bmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:256 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_51: "f32[256, 128, 4, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:260 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_37: "f32[256, 128, 4, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_107: "f32[256, 128, 256]" = torch.ops.aten.view.default(clone_37, [256, 128, 256]);  clone_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:262 in forward, code: attn_output = self.out_proj(attn_output)
        view_108: "f32[32768, 256]" = torch.ops.aten.view.default(view_107, [32768, 256]);  view_107 = None
        permute_52: "f32[256, 256]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_27: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg74_1, view_108, permute_52);  arg74_1 = view_108 = permute_52 = None
        view_109: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_27, [256, 128, 256]);  addmm_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:341 in forward, code: hidden_states = residual + hidden_states
        add_33: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_31, view_109);  add_31 = view_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:342 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16: "f32[256, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[256, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_23: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = rsqrt_8 = None
        mul_24: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_23, arg75_1);  mul_23 = arg75_1 = None
        add_35: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_24, arg76_1);  mul_24 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:369 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_110: "f32[32768, 256]" = torch.ops.aten.view.default(add_35, [32768, 256])
        permute_53: "f32[256, 2048]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_28: "f32[32768, 2048]" = torch.ops.aten.addmm.default(arg78_1, view_110, permute_53);  arg78_1 = view_110 = permute_53 = None
        view_111: "f32[256, 128, 2048]" = torch.ops.aten.view.default(addmm_28, [256, 128, 2048]);  addmm_28 = None
        relu_4: "f32[256, 128, 2048]" = torch.ops.aten.relu.default(view_111);  view_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:371 in forward, code: hidden_states = self.fc2(hidden_states)
        view_112: "f32[32768, 2048]" = torch.ops.aten.view.default(relu_4, [32768, 2048]);  relu_4 = None
        permute_54: "f32[2048, 256]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_29: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg80_1, view_112, permute_54);  arg80_1 = view_112 = permute_54 = None
        view_113: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_29, [256, 128, 256]);  addmm_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:373 in forward, code: hidden_states = residual + hidden_states
        add_36: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_35, view_113);  add_35 = view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:374 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_18: "f32[256, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[256, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_37: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_14: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_36, getitem_19);  add_36 = getitem_19 = None
        mul_25: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_26: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_25, arg81_1);  mul_25 = arg81_1 = None
        add_38: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_26, arg82_1);  mul_26 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:164 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_114: "f32[32768, 256]" = torch.ops.aten.view.default(add_38, [32768, 256])
        permute_55: "f32[256, 256]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_30: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg84_1, view_114, permute_55);  arg84_1 = view_114 = permute_55 = None
        view_115: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_30, [256, 128, 256]);  addmm_30 = None
        mul_27: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(view_115, 0.125);  view_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:189 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_116: "f32[32768, 256]" = torch.ops.aten.view.default(add_38, [32768, 256])
        permute_56: "f32[256, 256]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_31: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg86_1, view_116, permute_56);  arg86_1 = view_116 = permute_56 = None
        view_117: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_31, [256, 128, 256]);  addmm_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_118: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_117, [256, -1, 4, 64]);  view_117 = None
        permute_57: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_41: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:190 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_119: "f32[32768, 256]" = torch.ops.aten.view.default(add_38, [32768, 256])
        permute_58: "f32[256, 256]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_32: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg88_1, view_119, permute_58);  arg88_1 = view_119 = permute_58 = None
        view_120: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_32, [256, 128, 256]);  addmm_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_121: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(view_120, [256, -1, 4, 64]);  view_120 = None
        permute_59: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_42: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:144 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_122: "f32[256, 128, 4, 64]" = torch.ops.aten.view.default(mul_27, [256, 128, 4, 64]);  mul_27 = None
        permute_60: "f32[256, 4, 128, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        clone_43: "f32[256, 4, 128, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:203 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_123: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_43, [1024, -1, 64]);  clone_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:204 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_124: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_41, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:205 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_125: "f32[1024, 128, 64]" = torch.ops.aten.view.default(clone_42, [1024, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:208 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_61: "f32[1024, 64, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        bmm_10: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_123, permute_61);  view_123 = permute_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:221 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_126: "f32[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_10, [256, 4, 128, 128]);  bmm_10 = None
        add_39: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_126, expand_1);  view_126 = expand_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:222 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_127: "f32[1024, 128, 128]" = torch.ops.aten.view.default(add_39, [1024, 128, 128]);  add_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:224 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_5: "f32[1024, 128, 1]" = torch.ops.aten.amax.default(view_127, [-1], True)
        sub_15: "f32[1024, 128, 128]" = torch.ops.aten.sub.Tensor(view_127, amax_5);  view_127 = amax_5 = None
        exp_5: "f32[1024, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_6: "f32[1024, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[1024, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:247 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_11: "f32[1024, 128, 64]" = torch.ops.aten.bmm.default(div_5, view_125);  div_5 = view_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:255 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_128: "f32[256, 4, 128, 64]" = torch.ops.aten.view.default(bmm_11, [256, 4, 128, 64]);  bmm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:256 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_62: "f32[256, 128, 4, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:260 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_45: "f32[256, 128, 4, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_129: "f32[256, 128, 256]" = torch.ops.aten.view.default(clone_45, [256, 128, 256]);  clone_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:262 in forward, code: attn_output = self.out_proj(attn_output)
        view_130: "f32[32768, 256]" = torch.ops.aten.view.default(view_129, [32768, 256]);  view_129 = None
        permute_63: "f32[256, 256]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_33: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg90_1, view_130, permute_63);  arg90_1 = view_130 = permute_63 = None
        view_131: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_33, [256, 128, 256]);  addmm_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:341 in forward, code: hidden_states = residual + hidden_states
        add_40: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_38, view_131);  add_38 = view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:342 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_20: "f32[256, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[256, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_41: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_16: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_40, getitem_21);  add_40 = getitem_21 = None
        mul_28: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = rsqrt_10 = None
        mul_29: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_28, arg91_1);  mul_28 = arg91_1 = None
        add_42: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_29, arg92_1);  mul_29 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:369 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_132: "f32[32768, 256]" = torch.ops.aten.view.default(add_42, [32768, 256])
        permute_64: "f32[256, 2048]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_34: "f32[32768, 2048]" = torch.ops.aten.addmm.default(arg94_1, view_132, permute_64);  arg94_1 = view_132 = permute_64 = None
        view_133: "f32[256, 128, 2048]" = torch.ops.aten.view.default(addmm_34, [256, 128, 2048]);  addmm_34 = None
        relu_5: "f32[256, 128, 2048]" = torch.ops.aten.relu.default(view_133);  view_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:371 in forward, code: hidden_states = self.fc2(hidden_states)
        view_134: "f32[32768, 2048]" = torch.ops.aten.view.default(relu_5, [32768, 2048]);  relu_5 = None
        permute_65: "f32[2048, 256]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_35: "f32[32768, 256]" = torch.ops.aten.addmm.default(arg96_1, view_134, permute_65);  arg96_1 = view_134 = permute_65 = None
        view_135: "f32[256, 128, 256]" = torch.ops.aten.view.default(addmm_35, [256, 128, 256]);  addmm_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:373 in forward, code: hidden_states = residual + hidden_states
        add_43: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(add_42, view_135);  add_42 = view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:374 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_22: "f32[256, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[256, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_44: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_17: "f32[256, 128, 256]" = torch.ops.aten.sub.Tensor(add_43, getitem_23);  add_43 = getitem_23 = None
        mul_30: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_31: "f32[256, 128, 256]" = torch.ops.aten.mul.Tensor(mul_30, arg97_1);  mul_30 = arg97_1 = None
        add_45: "f32[256, 128, 256]" = torch.ops.aten.add.Tensor(mul_31, arg98_1);  mul_31 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:870 in forward, code: logits = self.lm_head(outputs[0])
        permute_66: "f32[256, 10000]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view_136: "f32[32768, 256]" = torch.ops.aten.view.default(add_45, [32768, 256]);  add_45 = None
        mm: "f32[32768, 10000]" = torch.ops.aten.mm.default(view_136, permute_66);  view_136 = permute_66 = None
        view_137: "f32[256, 128, 10000]" = torch.ops.aten.view.default(mm, [256, 128, 10000]);  mm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py:875 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_138: "f32[32768, 10000]" = torch.ops.aten.view.default(view_137, [-1, 10000])
        view_139: "i64[32768]" = torch.ops.aten.view.default(arg99_1, [-1]);  arg99_1 = None
        amax_6: "f32[32768, 1]" = torch.ops.aten.amax.default(view_138, [1], True)
        sub_18: "f32[32768, 10000]" = torch.ops.aten.sub.Tensor(view_138, amax_6);  view_138 = amax_6 = None
        exp_6: "f32[32768, 10000]" = torch.ops.aten.exp.default(sub_18)
        sum_7: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_19: "f32[32768, 10000]" = torch.ops.aten.sub.Tensor(sub_18, log);  sub_18 = log = None
        ne_1: "b8[32768]" = torch.ops.aten.ne.Scalar(view_139, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[32768]" = torch.ops.aten.where.self(ne_1, view_139, full_default_2);  ne_1 = full_default_2 = None
        unsqueeze_4: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[32768, 1]" = torch.ops.aten.gather.default(sub_19, 1, unsqueeze_4);  sub_19 = unsqueeze_4 = None
        squeeze: "f32[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_2: "b8[32768]" = torch.ops.aten.ne.Scalar(view_139, -100)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[32768]" = torch.ops.aten.where.self(ne_2, neg, full_default_3);  ne_2 = neg = full_default_3 = None
        ne_3: "b8[32768]" = torch.ops.aten.ne.Scalar(view_139, -100);  view_139 = None
        sum_8: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_3: "f32[]" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None
        sum_9: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_6: "f32[]" = torch.ops.aten.div.Tensor(sum_9, convert_element_type_3);  sum_9 = convert_element_type_3 = None
        return (div_6, view_137, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42)
        