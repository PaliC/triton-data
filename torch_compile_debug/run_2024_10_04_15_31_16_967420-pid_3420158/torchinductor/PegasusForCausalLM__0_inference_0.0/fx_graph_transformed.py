class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[32, 128]", arg1_1: "f32[50265, 1024]", arg2_1: "f32[1024, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024, 1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg8_1: "f32[1024]", arg9_1: "f32[1024, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[1024, 1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024]", arg14_1: "f32[1024]", arg15_1: "f32[4096, 1024]", arg16_1: "f32[4096]", arg17_1: "f32[1024, 4096]", arg18_1: "f32[1024]", arg19_1: "f32[1024]", arg20_1: "f32[1024]", arg21_1: "f32[1024, 1024]", arg22_1: "f32[1024]", arg23_1: "f32[1024, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024, 1024]", arg26_1: "f32[1024]", arg27_1: "f32[1024, 1024]", arg28_1: "f32[1024]", arg29_1: "f32[1024]", arg30_1: "f32[1024]", arg31_1: "f32[4096, 1024]", arg32_1: "f32[4096]", arg33_1: "f32[1024, 4096]", arg34_1: "f32[1024]", arg35_1: "f32[1024]", arg36_1: "f32[1024]", arg37_1: "f32[1024, 1024]", arg38_1: "f32[1024]", arg39_1: "f32[1024, 1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024, 1024]", arg42_1: "f32[1024]", arg43_1: "f32[1024, 1024]", arg44_1: "f32[1024]", arg45_1: "f32[1024]", arg46_1: "f32[1024]", arg47_1: "f32[4096, 1024]", arg48_1: "f32[4096]", arg49_1: "f32[1024, 4096]", arg50_1: "f32[1024]", arg51_1: "f32[1024]", arg52_1: "f32[1024]", arg53_1: "f32[1024, 1024]", arg54_1: "f32[1024]", arg55_1: "f32[1024, 1024]", arg56_1: "f32[1024]", arg57_1: "f32[1024, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024, 1024]", arg60_1: "f32[1024]", arg61_1: "f32[1024]", arg62_1: "f32[1024]", arg63_1: "f32[4096, 1024]", arg64_1: "f32[4096]", arg65_1: "f32[1024, 4096]", arg66_1: "f32[1024]", arg67_1: "f32[1024]", arg68_1: "f32[1024]", arg69_1: "f32[1024, 1024]", arg70_1: "f32[1024]", arg71_1: "f32[1024, 1024]", arg72_1: "f32[1024]", arg73_1: "f32[1024, 1024]", arg74_1: "f32[1024]", arg75_1: "f32[1024, 1024]", arg76_1: "f32[1024]", arg77_1: "f32[1024]", arg78_1: "f32[1024]", arg79_1: "f32[4096, 1024]", arg80_1: "f32[4096]", arg81_1: "f32[1024, 4096]", arg82_1: "f32[1024]", arg83_1: "f32[1024]", arg84_1: "f32[1024]", arg85_1: "f32[1024, 1024]", arg86_1: "f32[1024]", arg87_1: "f32[1024, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 1024]", arg90_1: "f32[1024]", arg91_1: "f32[1024, 1024]", arg92_1: "f32[1024]", arg93_1: "f32[1024]", arg94_1: "f32[1024]", arg95_1: "f32[4096, 1024]", arg96_1: "f32[4096]", arg97_1: "f32[1024, 4096]", arg98_1: "f32[1024]", arg99_1: "f32[1024]", arg100_1: "f32[1024]", arg101_1: "f32[1024, 1024]", arg102_1: "f32[1024]", arg103_1: "f32[1024, 1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024, 1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024, 1024]", arg108_1: "f32[1024]", arg109_1: "f32[1024]", arg110_1: "f32[1024]", arg111_1: "f32[4096, 1024]", arg112_1: "f32[4096]", arg113_1: "f32[1024, 4096]", arg114_1: "f32[1024]", arg115_1: "f32[1024]", arg116_1: "f32[1024]", arg117_1: "f32[1024, 1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024, 1024]", arg122_1: "f32[1024]", arg123_1: "f32[1024, 1024]", arg124_1: "f32[1024]", arg125_1: "f32[1024]", arg126_1: "f32[1024]", arg127_1: "f32[4096, 1024]", arg128_1: "f32[4096]", arg129_1: "f32[1024, 4096]", arg130_1: "f32[1024]", arg131_1: "f32[1024]", arg132_1: "f32[1024]", arg133_1: "f32[1024, 1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024, 1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024, 1024]", arg140_1: "f32[1024]", arg141_1: "f32[1024]", arg142_1: "f32[1024]", arg143_1: "f32[4096, 1024]", arg144_1: "f32[4096]", arg145_1: "f32[1024, 4096]", arg146_1: "f32[1024]", arg147_1: "f32[1024]", arg148_1: "f32[1024]", arg149_1: "f32[1024, 1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024, 1024]", arg152_1: "f32[1024]", arg153_1: "f32[1024, 1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024, 1024]", arg156_1: "f32[1024]", arg157_1: "f32[1024]", arg158_1: "f32[1024]", arg159_1: "f32[4096, 1024]", arg160_1: "f32[4096]", arg161_1: "f32[1024, 4096]", arg162_1: "f32[1024]", arg163_1: "f32[1024]", arg164_1: "f32[1024]", arg165_1: "f32[1024, 1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024, 1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024, 1024]", arg172_1: "f32[1024]", arg173_1: "f32[1024]", arg174_1: "f32[1024]", arg175_1: "f32[4096, 1024]", arg176_1: "f32[4096]", arg177_1: "f32[1024, 4096]", arg178_1: "f32[1024]", arg179_1: "f32[1024]", arg180_1: "f32[1024]", arg181_1: "f32[1024, 1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024, 1024]", arg186_1: "f32[1024]", arg187_1: "f32[1024, 1024]", arg188_1: "f32[1024]", arg189_1: "f32[1024]", arg190_1: "f32[1024]", arg191_1: "f32[4096, 1024]", arg192_1: "f32[4096]", arg193_1: "f32[1024, 4096]", arg194_1: "f32[1024]", arg195_1: "f32[1024]", arg196_1: "f32[1024]", arg197_1: "i64[32, 128]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:971 in forward, code: input_ids = input_ids.view(-1, input_shape[-1])
        view: "i64[32, 128]" = torch.ops.aten.reshape.default(arg0_1, [-1, 128]);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:981 in forward, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embedding: "f32[32, 128, 1024]" = torch.ops.aten.embedding.default(arg1_1, view, 0);  view = None
        mul: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:100 in forward, code: positions = torch.arange(
        iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:103 in forward, code: return super().forward(positions)
        embedding_1: "f32[128, 1024]" = torch.ops.aten.embedding.default(arg2_1, iota_1);  arg2_1 = iota_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:997 in forward, code: hidden_states = inputs_embeds + positions
        add_1: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  getitem_1 = None
        add_2: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_2: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_3, [4096, 1024])
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        
        # No stacktrace found for following nodes
        mm_default_72: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_2, permute);  view_2 = permute = None
        add_tensor_71: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_72, arg6_1);  mm_default_72 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_3: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_71, [32, 128, 1024]);  add_tensor_71 = None
        mul_3: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_10: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_3, [32, 128, 16, 64]);  mul_3 = None
        permute_5: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_11: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_3, [512, -1, 64]);  clone_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_4: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_3, [4096, 1024])
        permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        
        # No stacktrace found for following nodes
        mm_default_71: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
        add_tensor_70: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_71, arg8_1);  mm_default_71 = arg8_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_5: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_70, [32, 128, 1024]);  add_tensor_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_6: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_5, [32, -1, 16, 64]);  view_5 = None
        permute_2: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone_1: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_12: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_1, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_6: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_14: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [32, 16, 128, 128]);  bmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:159 in _make_causal_mask, code: mask_cond = torch.arange(mask.size(-1), device=device)
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:160 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        add: "i64[128]" = torch.ops.aten.add.Tensor(iota, 1)
        view_1: "i64[128, 1]" = torch.ops.aten.reshape.default(add, [128, 1]);  add = None
        lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:158 in _make_causal_mask, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        full_default: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:160 in _make_causal_mask, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        where: "f32[128, 128]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_3: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
        expand_1: "f32[32, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_3, [32, 1, 128, 128]);  unsqueeze_3 = None
        add_4: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_14, expand_1);  view_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_15: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_4, [512, 128, 128]);  add_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_15, [-1], True)
        sub_1: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
        exp: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_7: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_3, [4096, 1024]);  add_3 = None
        permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        
        # No stacktrace found for following nodes
        mm_default_70: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_7, permute_3);  view_7 = permute_3 = None
        add_tensor_69: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_70, arg10_1);  mm_default_70 = arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_8: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_69, [32, 128, 1024]);  add_tensor_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_9: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_8, [32, -1, 16, 64]);  view_8 = None
        permute_4: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_2: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_13: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_2, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_1: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div, view_13);  div = view_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_16: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [32, 16, 128, 64]);  bmm_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_7: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_5: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_5, [32, 128, 1024]);  clone_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_18: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_17, [4096, 1024]);  view_17 = None
        permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        
        # No stacktrace found for following nodes
        mm_default_69: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_18, permute_8);  view_18 = permute_8 = None
        add_tensor_68: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_69, arg12_1);  mm_default_69 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_19: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_68, [32, 128, 1024]);  add_tensor_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_5: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_1, view_19);  add_1 = view_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
        add_6: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_7: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_20: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_7, [4096, 1024]);  add_7 = None
        permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        
        # No stacktrace found for following nodes
        mm_default_68: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_20, permute_9);  view_20 = permute_9 = None
        add_tensor_67: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_68, arg16_1);  mm_default_68 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_21: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_67, [32, 128, 4096]);  add_tensor_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_6: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
        mul_7: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476);  view_21 = None
        erf: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_8: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_22: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_8, [4096, 4096]);  mul_8 = None
        permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        
        # No stacktrace found for following nodes
        mm_default_67: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_22, permute_10);  view_22 = permute_10 = None
        add_tensor_66: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_67, arg18_1);  mm_default_67 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_23: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_66, [32, 128, 1024]);  add_tensor_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_9: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_5, view_23);  add_5 = view_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  getitem_5 = None
        add_10: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
        add_11: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_24: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_11, [4096, 1024])
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        
        # No stacktrace found for following nodes
        mm_default_66: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_24, permute_11);  view_24 = permute_11 = None
        add_tensor_65: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_66, arg22_1);  mm_default_66 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_25: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_65, [32, 128, 1024]);  add_tensor_65 = None
        mul_11: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_25, 0.125);  view_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_32: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_11, [32, 128, 16, 64]);  mul_11 = None
        permute_16: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_11: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_33: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_11, [512, -1, 64]);  clone_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_26: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_11, [4096, 1024])
        permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        
        # No stacktrace found for following nodes
        mm_default_65: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_26, permute_12);  view_26 = permute_12 = None
        add_tensor_64: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_65, arg24_1);  mm_default_65 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_27: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_64, [32, 128, 1024]);  add_tensor_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_28: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_27, [32, -1, 16, 64]);  view_27 = None
        permute_13: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_9: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_34: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_9, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_17: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        bmm_2: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_33, permute_17);  view_33 = permute_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_36: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [32, 16, 128, 128]);  bmm_2 = None
        add_12: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_36, expand_1);  view_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_37: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_12, [512, 128, 128]);  add_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_1: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_37, [-1], True)
        sub_4: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_37, amax_1);  view_37 = amax_1 = None
        exp_1: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_29: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_11, [4096, 1024]);  add_11 = None
        permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        
        # No stacktrace found for following nodes
        mm_default_64: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_29, permute_14);  view_29 = permute_14 = None
        add_tensor_63: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_64, arg26_1);  mm_default_64 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_30: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_63, [32, 128, 1024]);  add_tensor_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_31: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_30, [32, -1, 16, 64]);  view_30 = None
        permute_15: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
        clone_10: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_35: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_10, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_3: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_1, view_35);  div_1 = view_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_38: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_3, [32, 16, 128, 64]);  bmm_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_18: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_13: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_13, [32, 128, 1024]);  clone_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_40: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_39, [4096, 1024]);  view_39 = None
        permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        
        # No stacktrace found for following nodes
        mm_default_63: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_40, permute_19);  view_40 = permute_19 = None
        add_tensor_62: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_63, arg28_1);  mm_default_63 = arg28_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_41: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_62, [32, 128, 1024]);  add_tensor_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_13: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_9, view_41);  add_9 = view_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  getitem_7 = None
        add_14: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        mul_12: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
        add_15: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_42: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_15, [4096, 1024]);  add_15 = None
        permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        
        # No stacktrace found for following nodes
        mm_default_62: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_42, permute_20);  view_42 = permute_20 = None
        add_tensor_61: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_62, arg32_1);  mm_default_62 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_43: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_61, [32, 128, 4096]);  add_tensor_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_14: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        mul_15: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
        erf_1: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_16: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_14, add_16);  mul_14 = add_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_44: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_16, [4096, 4096]);  mul_16 = None
        permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        
        # No stacktrace found for following nodes
        mm_default_61: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_44, permute_21);  view_44 = permute_21 = None
        add_tensor_60: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_61, arg34_1);  mm_default_61 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_45: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_60, [32, 128, 1024]);  add_tensor_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_17: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_13, view_45);  add_13 = view_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  getitem_9 = None
        add_18: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_17: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_18: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
        add_19: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_46: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_19, [4096, 1024])
        permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        
        # No stacktrace found for following nodes
        mm_default_60: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_46, permute_22);  view_46 = permute_22 = None
        add_tensor_59: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_60, arg38_1);  mm_default_60 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_47: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_59, [32, 128, 1024]);  add_tensor_59 = None
        mul_19: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_47, 0.125);  view_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_54: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_19, [32, 128, 16, 64]);  mul_19 = None
        permute_27: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_19: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_55: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_19, [512, -1, 64]);  clone_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_48: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_19, [4096, 1024])
        permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        
        # No stacktrace found for following nodes
        mm_default_59: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_48, permute_23);  view_48 = permute_23 = None
        add_tensor_58: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_59, arg40_1);  mm_default_59 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_49: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_58, [32, 128, 1024]);  add_tensor_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_50: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_49, [32, -1, 16, 64]);  view_49 = None
        permute_24: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_17: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_56: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_17, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_28: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        bmm_4: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_55, permute_28);  view_55 = permute_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_58: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [32, 16, 128, 128]);  bmm_4 = None
        add_20: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_58, expand_1);  view_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_59: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_20, [512, 128, 128]);  add_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_2: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_59, [-1], True)
        sub_7: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_59, amax_2);  view_59 = amax_2 = None
        exp_2: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_51: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_19, [4096, 1024]);  add_19 = None
        permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        
        # No stacktrace found for following nodes
        mm_default_58: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_51, permute_25);  view_51 = permute_25 = None
        add_tensor_57: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_58, arg42_1);  mm_default_58 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_52: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_57, [32, 128, 1024]);  add_tensor_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_53: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_52, [32, -1, 16, 64]);  view_52 = None
        permute_26: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_18: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_57: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_18, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_5: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_2, view_57);  div_2 = view_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_60: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [32, 16, 128, 64]);  bmm_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_29: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_21: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_61: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_21, [32, 128, 1024]);  clone_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_62: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_61, [4096, 1024]);  view_61 = None
        permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        
        # No stacktrace found for following nodes
        mm_default_57: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_62, permute_30);  view_62 = permute_30 = None
        add_tensor_56: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_57, arg44_1);  mm_default_57 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_63: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_56, [32, 128, 1024]);  add_tensor_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_21: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_17, view_63);  add_17 = view_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  getitem_11 = None
        add_22: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_20: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_21: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
        add_23: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_64: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_23, [4096, 1024]);  add_23 = None
        permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        
        # No stacktrace found for following nodes
        mm_default_56: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_64, permute_31);  view_64 = permute_31 = None
        add_tensor_55: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_56, arg48_1);  mm_default_56 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_65: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_55, [32, 128, 4096]);  add_tensor_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_22: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        mul_23: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476);  view_65 = None
        erf_2: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_24: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_22, add_24);  mul_22 = add_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_66: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_24, [4096, 4096]);  mul_24 = None
        permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        
        # No stacktrace found for following nodes
        mm_default_55: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_66, permute_32);  view_66 = permute_32 = None
        add_tensor_54: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_55, arg50_1);  mm_default_55 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_67: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_54, [32, 128, 1024]);  add_tensor_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_25: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_21, view_67);  add_21 = view_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  getitem_13 = None
        add_26: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_25: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
        add_27: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_68: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_27, [4096, 1024])
        permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        
        # No stacktrace found for following nodes
        mm_default_54: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_68, permute_33);  view_68 = permute_33 = None
        add_tensor_53: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_54, arg54_1);  mm_default_54 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_69: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_53, [32, 128, 1024]);  add_tensor_53 = None
        mul_27: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_69, 0.125);  view_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_76: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_27, [32, 128, 16, 64]);  mul_27 = None
        permute_38: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_27: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_77: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_27, [512, -1, 64]);  clone_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_70: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_27, [4096, 1024])
        permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        
        # No stacktrace found for following nodes
        mm_default_53: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_70, permute_34);  view_70 = permute_34 = None
        add_tensor_52: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_53, arg56_1);  mm_default_53 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_71: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_52, [32, 128, 1024]);  add_tensor_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_72: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_71, [32, -1, 16, 64]);  view_71 = None
        permute_35: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_25: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_78: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_25, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_39: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        bmm_6: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_77, permute_39);  view_77 = permute_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_80: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [32, 16, 128, 128]);  bmm_6 = None
        add_28: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_80, expand_1);  view_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_81: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_28, [512, 128, 128]);  add_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_3: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_81, [-1], True)
        sub_10: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_81, amax_3);  view_81 = amax_3 = None
        exp_3: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_73: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_27, [4096, 1024]);  add_27 = None
        permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        
        # No stacktrace found for following nodes
        mm_default_52: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_73, permute_36);  view_73 = permute_36 = None
        add_tensor_51: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_52, arg58_1);  mm_default_52 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_74: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_51, [32, 128, 1024]);  add_tensor_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_75: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_74, [32, -1, 16, 64]);  view_74 = None
        permute_37: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        clone_26: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_79: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_26, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_7: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_3, view_79);  div_3 = view_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_82: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_7, [32, 16, 128, 64]);  bmm_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_40: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_29: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_83: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_29, [32, 128, 1024]);  clone_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_84: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_83, [4096, 1024]);  view_83 = None
        permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        
        # No stacktrace found for following nodes
        mm_default_51: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_84, permute_41);  view_84 = permute_41 = None
        add_tensor_50: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_51, arg60_1);  mm_default_51 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_85: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_50, [32, 128, 1024]);  add_tensor_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_29: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_25, view_85);  add_25 = view_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  getitem_15 = None
        add_30: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_28: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_29: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
        add_31: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_86: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_31, [4096, 1024]);  add_31 = None
        permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        
        # No stacktrace found for following nodes
        mm_default_50: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_86, permute_42);  view_86 = permute_42 = None
        add_tensor_49: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_50, arg64_1);  mm_default_50 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_87: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_49, [32, 128, 4096]);  add_tensor_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_30: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_31: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476);  view_87 = None
        erf_3: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_32: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, add_32);  mul_30 = add_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_88: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_32, [4096, 4096]);  mul_32 = None
        permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        
        # No stacktrace found for following nodes
        mm_default_49: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_88, permute_43);  view_88 = permute_43 = None
        add_tensor_48: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_49, arg66_1);  mm_default_49 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_89: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_48, [32, 128, 1024]);  add_tensor_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_33: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_29, view_89);  add_29 = view_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  getitem_17 = None
        add_34: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_33: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_34: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
        add_35: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_90: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_35, [4096, 1024])
        permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        
        # No stacktrace found for following nodes
        mm_default_48: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_90, permute_44);  view_90 = permute_44 = None
        add_tensor_47: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_48, arg70_1);  mm_default_48 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_91: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_47, [32, 128, 1024]);  add_tensor_47 = None
        mul_35: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_91, 0.125);  view_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_98: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_35, [32, 128, 16, 64]);  mul_35 = None
        permute_49: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_35: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_99: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_35, [512, -1, 64]);  clone_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_92: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_35, [4096, 1024])
        permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        
        # No stacktrace found for following nodes
        mm_default_47: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_92, permute_45);  view_92 = permute_45 = None
        add_tensor_46: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_47, arg72_1);  mm_default_47 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_93: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_46, [32, 128, 1024]);  add_tensor_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_94: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_93, [32, -1, 16, 64]);  view_93 = None
        permute_46: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_33: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_100: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_33, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_50: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        bmm_8: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_99, permute_50);  view_99 = permute_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_102: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [32, 16, 128, 128]);  bmm_8 = None
        add_36: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_102, expand_1);  view_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_103: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_36, [512, 128, 128]);  add_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_4: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_103, [-1], True)
        sub_13: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_103, amax_4);  view_103 = amax_4 = None
        exp_4: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_95: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_35, [4096, 1024]);  add_35 = None
        permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        
        # No stacktrace found for following nodes
        mm_default_46: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_95, permute_47);  view_95 = permute_47 = None
        add_tensor_45: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_46, arg74_1);  mm_default_46 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_96: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_45, [32, 128, 1024]);  add_tensor_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_97: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_96, [32, -1, 16, 64]);  view_96 = None
        permute_48: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        clone_34: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_101: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_34, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_9: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_4, view_101);  div_4 = view_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_104: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_9, [32, 16, 128, 64]);  bmm_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_51: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_37: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_105: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_37, [32, 128, 1024]);  clone_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_105, [4096, 1024]);  view_105 = None
        permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        
        # No stacktrace found for following nodes
        mm_default_45: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_106, permute_52);  view_106 = permute_52 = None
        add_tensor_44: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_45, arg76_1);  mm_default_45 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_107: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_44, [32, 128, 1024]);  add_tensor_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_37: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_33, view_107);  add_33 = view_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  getitem_19 = None
        add_38: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_36: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_37: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
        add_39: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_108: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_39, [4096, 1024]);  add_39 = None
        permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        
        # No stacktrace found for following nodes
        mm_default_44: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_108, permute_53);  view_108 = permute_53 = None
        add_tensor_43: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_44, arg80_1);  mm_default_44 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_109: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_43, [32, 128, 4096]);  add_tensor_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_38: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_39: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_4: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_40: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_38, add_40);  mul_38 = add_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_110: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_40, [4096, 4096]);  mul_40 = None
        permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        
        # No stacktrace found for following nodes
        mm_default_43: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_110, permute_54);  view_110 = permute_54 = None
        add_tensor_42: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_43, arg82_1);  mm_default_43 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_111: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_42, [32, 128, 1024]);  add_tensor_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_41: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_37, view_111);  add_37 = view_111 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  getitem_21 = None
        add_42: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_41: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_42: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
        add_43: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_112: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_43, [4096, 1024])
        permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        
        # No stacktrace found for following nodes
        mm_default_42: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_112, permute_55);  view_112 = permute_55 = None
        add_tensor_41: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_42, arg86_1);  mm_default_42 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_113: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_41, [32, 128, 1024]);  add_tensor_41 = None
        mul_43: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_113, 0.125);  view_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_120: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_43, [32, 128, 16, 64]);  mul_43 = None
        permute_60: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_43: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_121: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_43, [512, -1, 64]);  clone_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_114: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_43, [4096, 1024])
        permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        
        # No stacktrace found for following nodes
        mm_default_41: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_114, permute_56);  view_114 = permute_56 = None
        add_tensor_40: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_41, arg88_1);  mm_default_41 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_115: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_40, [32, 128, 1024]);  add_tensor_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_116: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_115, [32, -1, 16, 64]);  view_115 = None
        permute_57: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        clone_41: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_122: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_41, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_61: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        bmm_10: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_121, permute_61);  view_121 = permute_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_124: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [32, 16, 128, 128]);  bmm_10 = None
        add_44: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_124, expand_1);  view_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_125: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_44, [512, 128, 128]);  add_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_5: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_125, [-1], True)
        sub_16: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_125, amax_5);  view_125 = amax_5 = None
        exp_5: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_117: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_43, [4096, 1024]);  add_43 = None
        permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        
        # No stacktrace found for following nodes
        mm_default_40: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_117, permute_58);  view_117 = permute_58 = None
        add_tensor_39: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_40, arg90_1);  mm_default_40 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_118: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_39, [32, 128, 1024]);  add_tensor_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_119: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_118, [32, -1, 16, 64]);  view_118 = None
        permute_59: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        clone_42: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_123: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_42, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_11: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_5, view_123);  div_5 = view_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_126: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_11, [32, 16, 128, 64]);  bmm_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_62: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_45: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_127: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_45, [32, 128, 1024]);  clone_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_128: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_127, [4096, 1024]);  view_127 = None
        permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        
        # No stacktrace found for following nodes
        mm_default_39: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_128, permute_63);  view_128 = permute_63 = None
        add_tensor_38: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_39, arg92_1);  mm_default_39 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_129: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_38, [32, 128, 1024]);  add_tensor_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_45: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_41, view_129);  add_41 = view_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  getitem_23 = None
        add_46: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_44: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_45: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
        add_47: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_130: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_47, [4096, 1024]);  add_47 = None
        permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        
        # No stacktrace found for following nodes
        mm_default_38: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_130, permute_64);  view_130 = permute_64 = None
        add_tensor_37: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_38, arg96_1);  mm_default_38 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_131: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_37, [32, 128, 4096]);  add_tensor_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_46: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        mul_47: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476);  view_131 = None
        erf_5: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_132: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_48, [4096, 4096]);  mul_48 = None
        permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        
        # No stacktrace found for following nodes
        mm_default_37: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_132, permute_65);  view_132 = permute_65 = None
        add_tensor_36: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_37, arg98_1);  mm_default_37 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_133: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_36, [32, 128, 1024]);  add_tensor_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_49: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_45, view_133);  add_45 = view_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  getitem_25 = None
        add_50: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_49: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_50: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
        add_51: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_134: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_51, [4096, 1024])
        permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        
        # No stacktrace found for following nodes
        mm_default_36: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_134, permute_66);  view_134 = permute_66 = None
        add_tensor_35: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_36, arg102_1);  mm_default_36 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_135: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_35, [32, 128, 1024]);  add_tensor_35 = None
        mul_51: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_135, 0.125);  view_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_142: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_51, [32, 128, 16, 64]);  mul_51 = None
        permute_71: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_51: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_143: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_51, [512, -1, 64]);  clone_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_136: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_51, [4096, 1024])
        permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        
        # No stacktrace found for following nodes
        mm_default_35: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_136, permute_67);  view_136 = permute_67 = None
        add_tensor_34: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_35, arg104_1);  mm_default_35 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_137: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_34, [32, 128, 1024]);  add_tensor_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_138: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_137, [32, -1, 16, 64]);  view_137 = None
        permute_68: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_49: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_144: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_49, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_72: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        bmm_12: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_143, permute_72);  view_143 = permute_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_146: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [32, 16, 128, 128]);  bmm_12 = None
        add_52: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_146, expand_1);  view_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_147: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_52, [512, 128, 128]);  add_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_6: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_147, [-1], True)
        sub_19: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_147, amax_6);  view_147 = amax_6 = None
        exp_6: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_139: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_51, [4096, 1024]);  add_51 = None
        permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        
        # No stacktrace found for following nodes
        mm_default_34: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_139, permute_69);  view_139 = permute_69 = None
        add_tensor_33: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_34, arg106_1);  mm_default_34 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_140: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_33, [32, 128, 1024]);  add_tensor_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_141: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_140, [32, -1, 16, 64]);  view_140 = None
        permute_70: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        clone_50: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_145: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_50, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_13: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_6, view_145);  div_6 = view_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_148: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_13, [32, 16, 128, 64]);  bmm_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_73: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_53: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_149: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_53, [32, 128, 1024]);  clone_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_150: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_149, [4096, 1024]);  view_149 = None
        permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        
        # No stacktrace found for following nodes
        mm_default_33: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_150, permute_74);  view_150 = permute_74 = None
        add_tensor_32: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_33, arg108_1);  mm_default_33 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_151: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_32, [32, 128, 1024]);  add_tensor_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_53: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_49, view_151);  add_49 = view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  getitem_27 = None
        add_54: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_52: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_53: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
        add_55: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_152: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_55, [4096, 1024]);  add_55 = None
        permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        
        # No stacktrace found for following nodes
        mm_default_32: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_152, permute_75);  view_152 = permute_75 = None
        add_tensor_31: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_32, arg112_1);  mm_default_32 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_153: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_31, [32, 128, 4096]);  add_tensor_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_54: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
        mul_55: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476);  view_153 = None
        erf_6: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_56: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_56);  mul_54 = add_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_154: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_56, [4096, 4096]);  mul_56 = None
        permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        
        # No stacktrace found for following nodes
        mm_default_31: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_154, permute_76);  view_154 = permute_76 = None
        add_tensor_30: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_31, arg114_1);  mm_default_31 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_155: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_30, [32, 128, 1024]);  add_tensor_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_57: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_53, view_155);  add_53 = view_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 128, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_21: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  getitem_29 = None
        add_58: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_57: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_58: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg115_1);  mul_57 = arg115_1 = None
        add_59: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_156: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_59, [4096, 1024])
        permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        
        # No stacktrace found for following nodes
        mm_default_30: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_156, permute_77);  view_156 = permute_77 = None
        add_tensor_29: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_30, arg118_1);  mm_default_30 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_157: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_29, [32, 128, 1024]);  add_tensor_29 = None
        mul_59: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_157, 0.125);  view_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_164: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_59, [32, 128, 16, 64]);  mul_59 = None
        permute_82: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        clone_59: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_165: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_59, [512, -1, 64]);  clone_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_158: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_59, [4096, 1024])
        permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        
        # No stacktrace found for following nodes
        mm_default_29: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_158, permute_78);  view_158 = permute_78 = None
        add_tensor_28: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_29, arg120_1);  mm_default_29 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_159: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_28, [32, 128, 1024]);  add_tensor_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_160: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_159, [32, -1, 16, 64]);  view_159 = None
        permute_79: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_57: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_166: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_57, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_83: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        bmm_14: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_165, permute_83);  view_165 = permute_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_168: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [32, 16, 128, 128]);  bmm_14 = None
        add_60: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_168, expand_1);  view_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_169: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_60, [512, 128, 128]);  add_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_7: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_169, [-1], True)
        sub_22: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_169, amax_7);  view_169 = amax_7 = None
        exp_7: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_161: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_59, [4096, 1024]);  add_59 = None
        permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        
        # No stacktrace found for following nodes
        mm_default_28: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_161, permute_80);  view_161 = permute_80 = None
        add_tensor_27: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_28, arg122_1);  mm_default_28 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_162: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_27, [32, 128, 1024]);  add_tensor_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_163: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_162, [32, -1, 16, 64]);  view_162 = None
        permute_81: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        clone_58: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_167: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_58, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_15: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_7, view_167);  div_7 = view_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_170: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_15, [32, 16, 128, 64]);  bmm_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_84: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_61: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_171: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_61, [32, 128, 1024]);  clone_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_172: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_171, [4096, 1024]);  view_171 = None
        permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        
        # No stacktrace found for following nodes
        mm_default_27: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_172, permute_85);  view_172 = permute_85 = None
        add_tensor_26: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_27, arg124_1);  mm_default_27 = arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_173: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_26, [32, 128, 1024]);  add_tensor_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_61: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_57, view_173);  add_57 = view_173 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 128, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  getitem_31 = None
        add_62: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_60: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_61: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_60, arg125_1);  mul_60 = arg125_1 = None
        add_63: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_174: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_63, [4096, 1024]);  add_63 = None
        permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        
        # No stacktrace found for following nodes
        mm_default_26: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_174, permute_86);  view_174 = permute_86 = None
        add_tensor_25: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_26, arg128_1);  mm_default_26 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_175: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_25, [32, 128, 4096]);  add_tensor_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_62: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_63: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476);  view_175 = None
        erf_7: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_64: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_62, add_64);  mul_62 = add_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_176: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_64, [4096, 4096]);  mul_64 = None
        permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        
        # No stacktrace found for following nodes
        mm_default_25: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_176, permute_87);  view_176 = permute_87 = None
        add_tensor_24: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_25, arg130_1);  mm_default_25 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_177: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_24, [32, 128, 1024]);  add_tensor_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_65: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_61, view_177);  add_61 = view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 128, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_24: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  getitem_33 = None
        add_66: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_65: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_66: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
        add_67: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_178: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_67, [4096, 1024])
        permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        
        # No stacktrace found for following nodes
        mm_default_24: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_178, permute_88);  view_178 = permute_88 = None
        add_tensor_23: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_24, arg134_1);  mm_default_24 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_179: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_23, [32, 128, 1024]);  add_tensor_23 = None
        mul_67: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_179, 0.125);  view_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_186: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_67, [32, 128, 16, 64]);  mul_67 = None
        permute_93: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_67: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_187: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_67, [512, -1, 64]);  clone_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_180: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_67, [4096, 1024])
        permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_180, permute_89);  view_180 = permute_89 = None
        add_tensor_22: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_23, arg136_1);  mm_default_23 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_181: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_22, [32, 128, 1024]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_182: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_181, [32, -1, 16, 64]);  view_181 = None
        permute_90: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_65: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_188: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_65, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_94: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        bmm_16: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_187, permute_94);  view_187 = permute_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_190: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [32, 16, 128, 128]);  bmm_16 = None
        add_68: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_190, expand_1);  view_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_191: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_68, [512, 128, 128]);  add_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_8: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_191, [-1], True)
        sub_25: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_191, amax_8);  view_191 = amax_8 = None
        exp_8: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_183: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_67, [4096, 1024]);  add_67 = None
        permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_183, permute_91);  view_183 = permute_91 = None
        add_tensor_21: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_22, arg138_1);  mm_default_22 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_184: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_21, [32, 128, 1024]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_185: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_184, [32, -1, 16, 64]);  view_184 = None
        permute_92: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_66: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_189: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_66, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_17: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_8, view_189);  div_8 = view_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_192: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_17, [32, 16, 128, 64]);  bmm_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_95: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_69: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_193: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_69, [32, 128, 1024]);  clone_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_194: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_193, [4096, 1024]);  view_193 = None
        permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_194, permute_96);  view_194 = permute_96 = None
        add_tensor_20: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_21, arg140_1);  mm_default_21 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_195: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_20, [32, 128, 1024]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_69: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_65, view_195);  add_65 = view_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 128, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  getitem_35 = None
        add_70: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_68: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_69: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_68, arg141_1);  mul_68 = arg141_1 = None
        add_71: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_196: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_71, [4096, 1024]);  add_71 = None
        permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_196, permute_97);  view_196 = permute_97 = None
        add_tensor_19: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_20, arg144_1);  mm_default_20 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_197: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_19, [32, 128, 4096]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_70: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
        mul_71: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476);  view_197 = None
        erf_8: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_72: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, add_72);  mul_70 = add_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_198: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_72, [4096, 4096]);  mul_72 = None
        permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_198, permute_98);  view_198 = permute_98 = None
        add_tensor_18: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_19, arg146_1);  mm_default_19 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_199: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_18, [32, 128, 1024]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_73: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_69, view_199);  add_69 = view_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 128, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_27: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  getitem_37 = None
        add_74: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_73: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_74: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg147_1);  mul_73 = arg147_1 = None
        add_75: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg148_1);  mul_74 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_200: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_75, [4096, 1024])
        permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_200, permute_99);  view_200 = permute_99 = None
        add_tensor_17: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_18, arg150_1);  mm_default_18 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_201: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_17, [32, 128, 1024]);  add_tensor_17 = None
        mul_75: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_201, 0.125);  view_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_208: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_75, [32, 128, 16, 64]);  mul_75 = None
        permute_104: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_75: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_209: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_75, [512, -1, 64]);  clone_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_202: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_75, [4096, 1024])
        permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_202, permute_100);  view_202 = permute_100 = None
        add_tensor_16: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_17, arg152_1);  mm_default_17 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_203: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_16, [32, 128, 1024]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_204: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_203, [32, -1, 16, 64]);  view_203 = None
        permute_101: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        clone_73: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_210: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_73, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_105: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        bmm_18: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_209, permute_105);  view_209 = permute_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_212: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [32, 16, 128, 128]);  bmm_18 = None
        add_76: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_212, expand_1);  view_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_213: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_76, [512, 128, 128]);  add_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_9: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_213, [-1], True)
        sub_28: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_213, amax_9);  view_213 = amax_9 = None
        exp_9: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_205: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_75, [4096, 1024]);  add_75 = None
        permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_205, permute_102);  view_205 = permute_102 = None
        add_tensor_15: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_16, arg154_1);  mm_default_16 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_206: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_15, [32, 128, 1024]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_207: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_206, [32, -1, 16, 64]);  view_206 = None
        permute_103: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
        clone_74: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_211: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_74, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_19: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_9, view_211);  div_9 = view_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_214: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_19, [32, 16, 128, 64]);  bmm_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_77: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_215: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_77, [32, 128, 1024]);  clone_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_216: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_215, [4096, 1024]);  view_215 = None
        permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_216, permute_107);  view_216 = permute_107 = None
        add_tensor_14: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_15, arg156_1);  mm_default_15 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_217: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_14, [32, 128, 1024]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_77: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_73, view_217);  add_73 = view_217 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 128, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  getitem_39 = None
        add_78: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_76: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_77: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_76, arg157_1);  mul_76 = arg157_1 = None
        add_79: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_77, arg158_1);  mul_77 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_218: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_79, [4096, 1024]);  add_79 = None
        permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_218, permute_108);  view_218 = permute_108 = None
        add_tensor_13: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_14, arg160_1);  mm_default_14 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_219: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_13, [32, 128, 4096]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_78: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        mul_79: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476);  view_219 = None
        erf_9: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_80: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_78, add_80);  mul_78 = add_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_220: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_80, [4096, 4096]);  mul_80 = None
        permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_220, permute_109);  view_220 = permute_109 = None
        add_tensor_12: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_13, arg162_1);  mm_default_13 = arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_221: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_12, [32, 128, 1024]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_81: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_77, view_221);  add_77 = view_221 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 128, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_30: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  getitem_41 = None
        add_82: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_81: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_82: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_81, arg163_1);  mul_81 = arg163_1 = None
        add_83: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_82, arg164_1);  mul_82 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_222: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_83, [4096, 1024])
        permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_222, permute_110);  view_222 = permute_110 = None
        add_tensor_11: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_12, arg166_1);  mm_default_12 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_223: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_11, [32, 128, 1024]);  add_tensor_11 = None
        mul_83: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_223, 0.125);  view_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_230: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_83, [32, 128, 16, 64]);  mul_83 = None
        permute_115: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
        clone_83: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_231: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_83, [512, -1, 64]);  clone_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_224: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_83, [4096, 1024])
        permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_224, permute_111);  view_224 = permute_111 = None
        add_tensor_10: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_11, arg168_1);  mm_default_11 = arg168_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_225: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_10, [32, 128, 1024]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_226: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_225, [32, -1, 16, 64]);  view_225 = None
        permute_112: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_81: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_232: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_81, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_116: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        bmm_20: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_231, permute_116);  view_231 = permute_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_234: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [32, 16, 128, 128]);  bmm_20 = None
        add_84: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_234, expand_1);  view_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_235: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_84, [512, 128, 128]);  add_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_10: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_235, [-1], True)
        sub_31: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_235, amax_10);  view_235 = amax_10 = None
        exp_10: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_227: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_83, [4096, 1024]);  add_83 = None
        permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_227, permute_113);  view_227 = permute_113 = None
        add_tensor_9: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_10, arg170_1);  mm_default_10 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_228: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_9, [32, 128, 1024]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_229: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_228, [32, -1, 16, 64]);  view_228 = None
        permute_114: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        clone_82: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_233: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_82, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_21: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_10, view_233);  div_10 = view_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_236: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_21, [32, 16, 128, 64]);  bmm_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_117: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_85: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_237: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_85, [32, 128, 1024]);  clone_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_238: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_237, [4096, 1024]);  view_237 = None
        permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_238, permute_118);  view_238 = permute_118 = None
        add_tensor_8: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_9, arg172_1);  mm_default_9 = arg172_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_239: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_8, [32, 128, 1024]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_85: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_81, view_239);  add_81 = view_239 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 128, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  getitem_43 = None
        add_86: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_84: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_85: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_84, arg173_1);  mul_84 = arg173_1 = None
        add_87: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_85, arg174_1);  mul_85 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_240: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_87, [4096, 1024]);  add_87 = None
        permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_240, permute_119);  view_240 = permute_119 = None
        add_tensor_7: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_8, arg176_1);  mm_default_8 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_241: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_7, [32, 128, 4096]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_86: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
        mul_87: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476);  view_241 = None
        erf_10: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_88: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_86, add_88);  mul_86 = add_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_242: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_88, [4096, 4096]);  mul_88 = None
        permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_242, permute_120);  view_242 = permute_120 = None
        add_tensor_6: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_7, arg178_1);  mm_default_7 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_243: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_6, [32, 128, 1024]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_89: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_85, view_243);  add_85 = view_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:401 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 128, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_33: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  getitem_45 = None
        add_90: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_89: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_90: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_89, arg179_1);  mul_89 = arg179_1 = None
        add_91: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_90, arg180_1);  mul_90 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_244: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_91, [4096, 1024])
        permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_244, permute_121);  view_244 = permute_121 = None
        add_tensor_5: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_6, arg182_1);  mm_default_6 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:162 in forward, code: query_states = self.q_proj(hidden_states) * self.scaling
        view_245: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_5, [32, 128, 1024]);  add_tensor_5 = None
        mul_91: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_245, 0.125);  view_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_252: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(mul_91, [32, 128, 16, 64]);  mul_91 = None
        permute_126: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        clone_91: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:201 in forward, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_253: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_91, [512, -1, 64]);  clone_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_246: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_91, [4096, 1024])
        permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_246, permute_122);  view_246 = permute_122 = None
        add_tensor_4: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_5, arg184_1);  mm_default_5 = arg184_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:187 in forward, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        view_247: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_4, [32, 128, 1024]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_248: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_247, [32, -1, 16, 64]);  view_247 = None
        permute_123: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        clone_89: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:202 in forward, code: key_states = key_states.reshape(*proj_shape)
        view_254: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_89, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:206 in forward, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_127: "f32[512, 64, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        bmm_22: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_253, permute_127);  view_253 = permute_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:219 in forward, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_256: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [32, 16, 128, 128]);  bmm_22 = None
        add_92: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_256, expand_1);  view_256 = expand_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:220 in forward, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_257: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(add_92, [512, 128, 128]);  add_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:222 in forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_11: "f32[512, 128, 1]" = torch.ops.aten.amax.default(view_257, [-1], True)
        sub_34: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(view_257, amax_11);  view_257 = amax_11 = None
        exp_11: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_249: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_91, [4096, 1024]);  add_91 = None
        permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_249, permute_124);  view_249 = permute_124 = None
        add_tensor_3: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_4, arg186_1);  mm_default_4 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:188 in forward, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        view_250: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, [32, 128, 1024]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:142 in _shape, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_251: "f32[32, 128, 16, 64]" = torch.ops.aten.reshape.default(view_250, [32, -1, 16, 64]);  view_250 = None
        permute_125: "f32[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        clone_90: "f32[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:203 in forward, code: value_states = value_states.reshape(*proj_shape)
        view_255: "f32[512, 128, 64]" = torch.ops.aten.reshape.default(clone_90, [512, -1, 64])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:245 in forward, code: attn_output = torch.bmm(attn_probs, value_states)
        bmm_23: "f32[512, 128, 64]" = torch.ops.aten.bmm.default(div_11, view_255);  div_11 = view_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:253 in forward, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_258: "f32[32, 16, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, [32, 16, 128, 64]);  bmm_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:254 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_128: "f32[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:258 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_93: "f32[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_259: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(clone_93, [32, 128, 1024]);  clone_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_260: "f32[4096, 1024]" = torch.ops.aten.reshape.default(view_259, [4096, 1024]);  view_259 = None
        permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_260, permute_129);  view_260 = permute_129 = None
        add_tensor_2: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_3, arg188_1);  mm_default_3 = arg188_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:260 in forward, code: attn_output = self.out_proj(attn_output)
        view_261: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor_2, [32, 128, 1024]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:415 in forward, code: hidden_states = residual + hidden_states
        add_93: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_89, view_261);  add_89 = view_261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:442 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 128, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  getitem_47 = None
        add_94: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_92: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_93: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_92, arg189_1);  mul_92 = arg189_1 = None
        add_95: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_93, arg190_1);  mul_93 = arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_262: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_95, [4096, 1024]);  add_95 = None
        permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[4096, 4096]" = torch.ops.aten.mm.default(view_262, permute_130);  view_262 = permute_130 = None
        add_tensor_1: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_default_2, arg192_1);  mm_default_2 = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:443 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_263: "f32[32, 128, 4096]" = torch.ops.aten.reshape.default(add_tensor_1, [32, 128, 4096]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_94: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
        mul_95: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476);  view_263 = None
        erf_11: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_96: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_94, add_96);  mul_94 = add_96 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_264: "f32[4096, 4096]" = torch.ops.aten.reshape.default(mul_96, [4096, 4096]);  mul_96 = None
        permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_264, permute_131);  view_264 = permute_131 = None
        add_tensor: "f32[4096, 1024]" = torch.ops.aten.add.Tensor(mm_default_1, arg194_1);  mm_default_1 = arg194_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:445 in forward, code: hidden_states = self.fc2(hidden_states)
        view_265: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(add_tensor, [32, 128, 1024]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:447 in forward, code: hidden_states = residual + hidden_states
        add_97: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_93, view_265);  add_93 = view_265 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1071 in forward, code: hidden_states = self.layer_norm(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 128, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1646 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_269: "i64[4096]" = torch.ops.aten.reshape.default(arg197_1, [-1]);  arg197_1 = None
        ne_1: "b8[4096]" = torch.ops.aten.ne.Scalar(view_269, -100)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1071 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_36: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        add_98: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_97: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_98: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_97, arg195_1);  mul_97 = arg195_1 = None
        add_99: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1640 in forward, code: logits = self.lm_head(outputs[0])
        view_266: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_99, [4096, 1024]);  add_99 = None
        permute_132: "f32[1024, 50265]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        
        # No stacktrace found for following nodes
        full_default_4: "f32[1024, 3]" = torch.ops.aten.full.default([1024, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[1024, 50268]" = torch.ops.aten.cat.default([permute_132, full_default_4], 1);  permute_132 = full_default_4 = None
        mm_default: "f32[4096, 50268]" = torch.ops.aten.mm.default(view_266, cat_default);  view_266 = cat_default = None
        slice_tensor: "f32[4096, 50265]" = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3);  mm_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1640 in forward, code: logits = self.lm_head(outputs[0])
        view_267: "f32[32, 128, 50265]" = torch.ops.aten.reshape.default(slice_tensor, [32, 128, 50265]);  slice_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/pegasus/modeling_pegasus.py:1646 in forward, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_268: "f32[4096, 50265]" = torch.ops.aten.reshape.default(view_267, [-1, 50265])
        amax_12: "f32[4096, 1]" = torch.ops.aten.amax.default(view_268, [1], True)
        sub_37: "f32[4096, 50265]" = torch.ops.aten.sub.Tensor(view_268, amax_12);  view_268 = amax_12 = None
        exp_12: "f32[4096, 50265]" = torch.ops.aten.exp.default(sub_37)
        sum_13: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_38: "f32[4096, 50265]" = torch.ops.aten.sub.Tensor(sub_37, log);  sub_37 = log = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_269, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[4096]" = torch.ops.aten.where.self(ne, view_269, full_default_2);  ne = full_default_2 = None
        unsqueeze_4: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_38, 1, unsqueeze_4);  sub_38 = unsqueeze_4 = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_2: "b8[4096]" = torch.ops.aten.ne.Scalar(view_269, -100);  view_269 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type);  sum_15 = convert_element_type = None
        return (div_12, view_267, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42, clone_49, clone_50, clone_57, clone_58, clone_65, clone_66, clone_73, clone_74, clone_81, clone_82, clone_89, clone_90)
        