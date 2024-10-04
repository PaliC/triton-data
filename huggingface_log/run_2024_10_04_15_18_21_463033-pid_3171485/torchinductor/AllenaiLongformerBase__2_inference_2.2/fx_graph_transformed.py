class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 1024, 768]", arg1_1: "f32[768, 768]", arg2_1: "f32[768]", arg3_1: "f32[768, 768]", arg4_1: "f32[768]", arg5_1: "f32[768, 768]", arg6_1: "f32[768]", arg7_1: "f32[4, 1024]", arg8_1: "b8[4, 1024]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[3072, 768]", arg14_1: "f32[3072]", arg15_1: "f32[768, 3072]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768, 768]", arg20_1: "f32[768]", arg21_1: "f32[768, 768]", arg22_1: "f32[768]", arg23_1: "f32[768, 768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[3072, 768]", arg30_1: "f32[3072]", arg31_1: "f32[768, 3072]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768, 768]", arg36_1: "f32[768]", arg37_1: "f32[768, 768]", arg38_1: "f32[768]", arg39_1: "f32[768, 768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[3072, 768]", arg46_1: "f32[3072]", arg47_1: "f32[768, 3072]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768, 768]", arg52_1: "f32[768]", arg53_1: "f32[768, 768]", arg54_1: "f32[768]", arg55_1: "f32[768, 768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[3072, 768]", arg62_1: "f32[3072]", arg63_1: "f32[768, 3072]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768, 768]", arg68_1: "f32[768]", arg69_1: "f32[768, 768]", arg70_1: "f32[768]", arg71_1: "f32[768, 768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[3072, 768]", arg78_1: "f32[3072]", arg79_1: "f32[768, 3072]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[768, 768]", arg84_1: "f32[768]", arg85_1: "f32[768, 768]", arg86_1: "f32[768]", arg87_1: "f32[768, 768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768]", arg92_1: "f32[768]", arg93_1: "f32[3072, 768]", arg94_1: "f32[3072]", arg95_1: "f32[768, 3072]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768, 768]", arg100_1: "f32[768]", arg101_1: "f32[768, 768]", arg102_1: "f32[768]", arg103_1: "f32[768, 768]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[3072, 768]", arg110_1: "f32[3072]", arg111_1: "f32[768, 3072]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[768, 768]", arg116_1: "f32[768]", arg117_1: "f32[768, 768]", arg118_1: "f32[768]", arg119_1: "f32[768, 768]", arg120_1: "f32[768]", arg121_1: "f32[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[3072, 768]", arg126_1: "f32[3072]", arg127_1: "f32[768, 3072]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[768, 768]", arg132_1: "f32[768]", arg133_1: "f32[768, 768]", arg134_1: "f32[768]", arg135_1: "f32[768, 768]", arg136_1: "f32[768]", arg137_1: "f32[768, 768]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[768]", arg141_1: "f32[3072, 768]", arg142_1: "f32[3072]", arg143_1: "f32[768, 3072]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768, 768]", arg148_1: "f32[768]", arg149_1: "f32[768, 768]", arg150_1: "f32[768]", arg151_1: "f32[768, 768]", arg152_1: "f32[768]", arg153_1: "f32[768, 768]", arg154_1: "f32[768]", arg155_1: "f32[768]", arg156_1: "f32[768]", arg157_1: "f32[3072, 768]", arg158_1: "f32[3072]", arg159_1: "f32[768, 3072]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[768, 768]", arg164_1: "f32[768]", arg165_1: "f32[768, 768]", arg166_1: "f32[768]", arg167_1: "f32[768, 768]", arg168_1: "f32[768]", arg169_1: "f32[768, 768]", arg170_1: "f32[768]", arg171_1: "f32[768]", arg172_1: "f32[768]", arg173_1: "f32[3072, 768]", arg174_1: "f32[3072]", arg175_1: "f32[768, 3072]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[768, 768]", arg180_1: "f32[768]", arg181_1: "f32[768, 768]", arg182_1: "f32[768]", arg183_1: "f32[768, 768]", arg184_1: "f32[768]", arg185_1: "f32[768, 768]", arg186_1: "f32[768]", arg187_1: "f32[768]", arg188_1: "f32[768]", arg189_1: "f32[3072, 768]", arg190_1: "f32[3072]", arg191_1: "f32[768, 3072]", arg192_1: "f32[768]", arg193_1: "f32[768]", arg194_1: "f32[768]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_15: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_16: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 3);  unsqueeze_15 = None
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_10: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1);  slice_10 = None
        slice_6: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_8: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_6, 3, 256, 9223372036854775807);  slice_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone, [4096, 768]);  clone = None
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm: "f32[4096, 768]" = torch.ops.aten.mm.default(view, permute_1);  view = permute_1 = None
        view_1: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm, [1024, 4, 768]);  mm = None
        add: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_1, arg2_1);  view_1 = arg2_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add, 8.0);  add = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_12: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div, [1024, 4, 12, 64]);  div = None
        permute_11: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_12, [1, 0, 2, 3]);  view_12 = None
        permute_12: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_11, [0, 2, 1, 3]);  permute_11 = None
        view_13: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_12, [48, 1024, 64]);  permute_12 = None
        view_14: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_13, [48, 2, 512, 64]);  view_13 = None
        as_strided_2: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_14, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_14 = None
        unsqueeze_2: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_2, 4);  as_strided_2 = None
        permute_13: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_2, [0, 1, 2, 4, 3]);  unsqueeze_2 = None
        permute_14: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_13, [0, 1, 2, 4, 3]);  permute_13 = None
        clone_3: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_15: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_3, [144, 512, 64]);  clone_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_1: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view_2: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_1, [4096, 768]);  clone_1 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm_1: "f32[4096, 768]" = torch.ops.aten.mm.default(view_2, permute_2);  view_2 = permute_2 = None
        view_3: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_1, [1024, 4, 768]);  mm_1 = None
        add_1: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_3, arg4_1);  view_3 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_7: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_1, [1024, 4, 12, 64]);  add_1 = None
        permute_5: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_7, [1, 0, 2, 3]);  view_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_7: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_9: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_7, [48, 1024, 64]);  permute_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_11: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_9, [48, 2, 512, 64]);  view_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_1: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_11, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_1: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_9: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_1, [0, 1, 4, 2, 3]);  unsqueeze_1 = None
        permute_15: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_9, [0, 1, 4, 3, 2]);  permute_9 = None
        clone_4: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_16: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_4, [144, 64, 512]);  clone_4 = None
        bmm: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
        view_17: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm, [48, 3, 512, 1, 512]);  bmm = None
        permute_16: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_17, [0, 1, 2, 4, 3]);  view_17 = None
        view_18: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_16, [48, 3, 512, 512]);  permute_16 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_18, [0, 0, 0, 1], 0.0);  view_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_19: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd, [48, 3, 512, 513]);  constant_pad_nd = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_3: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_19, 2, 0, 256)
        slice_4: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 257);  slice_3 = None
        copy: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_8, slice_4);  slice_8 = slice_4 = None
        
        # No stacktrace found for following nodes
        slice_tensor: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_scatter_default: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy, 3, 256, 9223372036854775807);  slice_tensor = copy = None
        slice_scatter_default_1: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_default, 1, 0, -1);  full = slice_scatter_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_3: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1);  select_3 = None
        select_2: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        slice_24: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_2, 2, 256, 9223372036854775807);  select_2 = None
        select: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_19, 1, -1)
        slice_17: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
        slice_18: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_17, 2, 0, 257);  slice_17 = None
        copy_1: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_24, slice_18);  slice_24 = slice_18 = None
        
        # No stacktrace found for following nodes
        select_int: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        slice_scatter_default_2: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int, copy_1, 2, 256, 9223372036854775807);  select_int = copy_1 = None
        select_scatter_default: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 1, -1);  slice_scatter_default_1 = slice_scatter_default_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_43: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807);  slice_43 = None
        slice_39: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_41: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 256);  slice_39 = None
        slice_32: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_19, 2, -257, -1)
        slice_33: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_32, 3, 257, 9223372036854775807);  slice_32 = None
        copy_2: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_41, slice_33);  slice_41 = slice_33 = None
        
        # No stacktrace found for following nodes
        slice_tensor_1: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default, 1, 1, 9223372036854775807)
        slice_scatter_default_3: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_1, copy_2, 3, 0, 256);  slice_tensor_1 = copy_2 = None
        slice_scatter_default_4: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default, slice_scatter_default_3, 1, 1, 9223372036854775807);  select_scatter_default = slice_scatter_default_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_8: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_59: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_8, 1, 1, 256);  select_8 = slice_59 = None
        select_7: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_56: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_7, 1, 1, 256);  select_7 = None
        slice_57: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_56, 2, 1, 256);  slice_56 = None
        select_5: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_19, 1, 0);  view_19 = None
        slice_50: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_5, 1, 0, 255);  select_5 = None
        slice_51: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_50, 2, -255, 9223372036854775807);  slice_50 = None
        copy_3: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_57, slice_51);  slice_57 = slice_51 = None
        
        # No stacktrace found for following nodes
        select_int_1: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_4, 1, 0)
        slice_tensor_2: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_1, 1, 1, 256)
        slice_scatter_default_5: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_2, copy_3, 2, 1, 256);  slice_tensor_2 = copy_3 = None
        slice_scatter_default_6: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_1, slice_scatter_default_5, 1, 1, 256);  select_int_1 = slice_scatter_default_5 = None
        select_scatter_default_1: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_4, slice_scatter_default_6, 1, 0);  slice_scatter_default_4 = slice_scatter_default_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_24: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_1, [4, 12, 1024, 513])
        permute_21: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        slice_82: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_21, 1, 0, 256);  slice_82 = None
        view_23: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_1, [4, 12, 1024, 513])
        permute_20: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        slice_78: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_20, 1, 0, 256);  permute_20 = None
        slice_80: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_78, 3, 0, 257);  slice_78 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_4);  unsqueeze_3 = unsqueeze_4 = None
        le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
        full_default: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[256, 257]" = torch.ops.aten.where.self(le, full_default, full_default_1);  le = full_default = full_default_1 = None
        rev: "f32[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_5: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_6: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_6, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_2: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_22: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_1, [4, 12, 1024, 513]);  select_scatter_default_1 = None
        permute_19: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        slice_70: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_19, 1, 0, 256);  permute_19 = None
        slice_72: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_70, 3, 0, 257);  slice_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_1: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_2, slice_72);  convert_element_type = full_default_2 = slice_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_4: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_80, where_1);  slice_80 = where_1 = None
        
        # No stacktrace found for following nodes
        slice_tensor_3: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_21, 1, 0, 256)
        slice_scatter_default_7: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_3, copy_4, 3, 0, 257);  slice_tensor_3 = copy_4 = None
        slice_scatter_default_8: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_21, slice_scatter_default_7, 1, 0, 256);  permute_21 = slice_scatter_default_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_22: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_8, [0, 2, 1, 3]);  slice_scatter_default_8 = None
        view_25: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_22, [48, 4, 256, 513]);  permute_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_29: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_25, [4, 12, 1024, 513])
        permute_26: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        slice_105: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_26, 1, -256, 9223372036854775807);  slice_105 = None
        view_28: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_25, [4, 12, 1024, 513])
        permute_25: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        slice_101: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_25, 1, -256, 9223372036854775807);  permute_25 = None
        slice_103: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_101, 3, -257, 9223372036854775807);  slice_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_1: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_6, [1, 3]);  unsqueeze_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_1: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, [4, 256, 12, 257]);  rev_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_1: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_3: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_27: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_25, [4, 12, 1024, 513]);  view_25 = None
        permute_24: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        slice_93: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_24, 1, -256, 9223372036854775807);  permute_24 = None
        slice_95: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_93, 3, -257, 9223372036854775807);  slice_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_2: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_3, slice_95);  convert_element_type_1 = full_default_3 = slice_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_5: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_103, where_2);  slice_103 = where_2 = None
        
        # No stacktrace found for following nodes
        slice_tensor_4: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_26, 1, -256, 9223372036854775807)
        slice_scatter_default_9: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_4, copy_5, 3, -257, 9223372036854775807);  slice_tensor_4 = copy_5 = None
        slice_scatter_default_10: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_26, slice_scatter_default_9, 1, -256, 9223372036854775807);  permute_26 = slice_scatter_default_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_27: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_10, [0, 2, 1, 3]);  slice_scatter_default_10 = None
        view_30: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_27, [48, 4, 256, 513]);  permute_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_50: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_30, [4, 12, 1024, 513]);  view_30 = None
        permute_45: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_5: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_122: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1);  slice_122 = None
        slice_118: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_120: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_118, 3, 256, 9223372036854775807);  slice_118 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_4: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_29: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_32: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_29, [4, 1024, 1]);  permute_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_34: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_32, [4, 2, 512, 1]);  view_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_3: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_34, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_9: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_3, 4);  as_strided_3 = None
        permute_31: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_9, [0, 1, 2, 4, 3]);  unsqueeze_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_7: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_8: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_2: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_8, torch.float32)
        where_3: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_8, full_default_4, convert_element_type_2);  unsqueeze_8 = full_default_4 = convert_element_type_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_30: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_33: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_30, [4, 1024, 1]);  permute_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_35: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_33, [4, 2, 512, 1]);  view_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_4: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_35, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_10: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_4, 4);  as_strided_4 = None
        permute_32: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_10, [0, 1, 4, 2, 3]);  unsqueeze_10 = None
        mul: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_31, permute_32);  permute_31 = permute_32 = None
        view_36: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul, [4, 3, 512, 512]);  mul = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_36, [0, 0, 0, 1], 0.0);  view_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_37: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_1, [4, 3, 512, 513]);  constant_pad_nd_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_115: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_37, 2, 0, 256)
        slice_116: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_115, 3, 0, 257);  slice_115 = None
        copy_6: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_120, slice_116);  slice_120 = slice_116 = None
        
        # No stacktrace found for following nodes
        slice_tensor_5: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_scatter_default_11: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_5, copy_6, 3, 256, 9223372036854775807);  slice_tensor_5 = copy_6 = None
        slice_scatter_default_12: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_default_11, 1, 0, -1);  full_5 = slice_scatter_default_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_13: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_12, 1, -1);  select_13 = None
        select_12: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_12, 1, -1)
        slice_136: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_12, 2, 256, 9223372036854775807);  select_12 = None
        select_10: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_37, 1, -1)
        slice_129: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_10, 1, 256, 9223372036854775807);  select_10 = None
        slice_130: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_129, 2, 0, 257);  slice_129 = None
        copy_7: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_136, slice_130);  slice_136 = slice_130 = None
        
        # No stacktrace found for following nodes
        select_int_2: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_12, 1, -1)
        slice_scatter_default_13: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_2, copy_7, 2, 256, 9223372036854775807);  select_int_2 = copy_7 = None
        select_scatter_default_2: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_12, slice_scatter_default_13, 1, -1);  slice_scatter_default_12 = slice_scatter_default_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_155: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 1, 9223372036854775807);  slice_155 = None
        slice_151: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 1, 9223372036854775807)
        slice_153: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_151, 3, 0, 256);  slice_151 = None
        slice_144: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_37, 2, -257, -1)
        slice_145: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_144, 3, 257, 9223372036854775807);  slice_144 = None
        copy_8: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_153, slice_145);  slice_153 = slice_145 = None
        
        # No stacktrace found for following nodes
        slice_tensor_6: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_2, 1, 1, 9223372036854775807)
        slice_scatter_default_14: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_6, copy_8, 3, 0, 256);  slice_tensor_6 = copy_8 = None
        slice_scatter_default_15: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_2, slice_scatter_default_14, 1, 1, 9223372036854775807);  select_scatter_default_2 = slice_scatter_default_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_18: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, 0)
        slice_171: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_18, 1, 1, 256);  select_18 = slice_171 = None
        select_17: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, 0)
        slice_168: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_17, 1, 1, 256);  select_17 = None
        slice_169: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_168, 2, 1, 256);  slice_168 = None
        select_15: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_37, 1, 0);  view_37 = None
        slice_162: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_15, 1, 0, 255);  select_15 = None
        slice_163: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_162, 2, -255, 9223372036854775807);  slice_162 = None
        copy_9: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_169, slice_163);  slice_169 = slice_163 = None
        
        # No stacktrace found for following nodes
        select_int_3: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_15, 1, 0)
        slice_tensor_7: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_3, 1, 1, 256)
        slice_scatter_default_16: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_7, copy_9, 2, 1, 256);  slice_tensor_7 = copy_9 = None
        slice_scatter_default_17: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_3, slice_scatter_default_16, 1, 1, 256);  select_int_3 = slice_scatter_default_16 = None
        select_scatter_default_3: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_15, slice_scatter_default_17, 1, 0);  slice_scatter_default_15 = slice_scatter_default_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_42: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_3, [4, 1, 1024, 513])
        permute_37: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        slice_194: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_37, 1, 0, 256);  slice_194 = None
        view_41: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_3, [4, 1, 1024, 513])
        permute_36: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        slice_190: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_36, 1, 0, 256);  permute_36 = None
        slice_192: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_190, 3, 0, 257);  slice_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_2: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_11: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_2, -2);  iota_2 = None
        iota_3: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_12: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
        sub_3: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_11, unsqueeze_12);  unsqueeze_11 = unsqueeze_12 = None
        le_1: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_3, 0);  sub_3 = None
        full_default_5: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[256, 257]" = torch.ops.aten.where.self(le_1, full_default_5, full_default_6);  le_1 = full_default_5 = full_default_6 = None
        rev_2: "f32[256, 257]" = torch.ops.prims.rev.default(where_4, [0]);  where_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_13: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_2, 0);  rev_2 = None
        unsqueeze_14: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_2: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_14, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_3: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_7: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_40: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_3, [4, 1, 1024, 513]);  select_scatter_default_3 = None
        permute_35: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        slice_182: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_35, 1, 0, 256);  permute_35 = None
        slice_184: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_182, 3, 0, 257);  slice_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_5: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_3, full_default_7, slice_184);  convert_element_type_3 = full_default_7 = slice_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_10: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_192, where_5);  slice_192 = where_5 = None
        
        # No stacktrace found for following nodes
        slice_tensor_8: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_37, 1, 0, 256)
        slice_scatter_default_18: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_8, copy_10, 3, 0, 257);  slice_tensor_8 = copy_10 = None
        slice_scatter_default_19: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_37, slice_scatter_default_18, 1, 0, 256);  permute_37 = slice_scatter_default_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_38: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_19, [0, 2, 1, 3]);  slice_scatter_default_19 = None
        view_43: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_38, [4, 4, 256, 513]);  permute_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_47: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_43, [4, 1, 1024, 513])
        permute_42: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        slice_217: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_42, 1, -256, 9223372036854775807);  slice_217 = None
        view_46: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_43, [4, 1, 1024, 513])
        permute_41: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        slice_213: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_41, 1, -256, 9223372036854775807);  permute_41 = None
        slice_215: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_213, 3, -257, 9223372036854775807);  slice_213 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_3: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_14, [1, 3]);  unsqueeze_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_3: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_3, [4, 256, 1, 257]);  rev_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_4: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_8: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_45: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_43, [4, 1, 1024, 513]);  view_43 = None
        permute_40: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        slice_205: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_40, 1, -256, 9223372036854775807);  permute_40 = None
        slice_207: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_205, 3, -257, 9223372036854775807);  slice_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_6: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_default_8, slice_207);  convert_element_type_4 = full_default_8 = slice_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_11: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_215, where_6);  slice_215 = where_6 = None
        
        # No stacktrace found for following nodes
        slice_tensor_9: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_42, 1, -256, 9223372036854775807)
        slice_scatter_default_20: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_9, copy_11, 3, -257, 9223372036854775807);  slice_tensor_9 = copy_11 = None
        slice_scatter_default_21: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_42, slice_scatter_default_20, 1, -256, 9223372036854775807);  permute_42 = slice_scatter_default_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_43: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_21, [0, 2, 1, 3]);  slice_scatter_default_21 = None
        view_48: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_43, [4, 4, 256, 513]);  permute_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_51: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_48, [4, 1, 1024, 513]);  view_48 = None
        permute_46: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        add_5: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_45, permute_46);  permute_45 = permute_46 = None
        permute_47: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_5, [0, 2, 1, 3]);  add_5 = None
        view_53: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_47, [48, 4, 256, 513]);  permute_47 = None
        view_54: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_53, [4, 12, 1024, 513]);  view_53 = None
        permute_48: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_5: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        amax: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_5, [-1], True)
        sub_4: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_5, amax);  clone_5 = amax = None
        exp: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_1: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_7: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_7: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_9, div_7);  unsqueeze_16 = full_default_9 = div_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_50: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_7, [0, 2, 1, 3]);  where_7 = None
        clone_7: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_56: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_7, [48, 4, 256, 513]);  clone_7 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_3: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_56, [0, 257], 0.0);  view_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_58: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_3, [48, 4, -1]);  constant_pad_nd_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_227: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_58, 2, 0, -256);  view_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_59: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_227, [48, 4, 256, 769]);  slice_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_231: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_59, 3, 0, -1);  view_59 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_17: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_231, 4);  slice_231 = None
        permute_52: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_17, [0, 1, 2, 4, 3]);  unsqueeze_17 = None
        permute_54: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_52, [0, 1, 2, 4, 3]);  permute_52 = None
        view_60: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_54, [192, 256, 768]);  permute_54 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_2: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_2, [4096, 768]);  clone_2 = None
        permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_2: "f32[4096, 768]" = torch.ops.aten.mm.default(view_4, permute_3);  view_4 = permute_3 = None
        view_5: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_2, [1024, 4, 768]);  mm_2 = None
        add_2: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_5, arg6_1);  view_5 = arg6_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_55: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_2, [1024, 4, 12, 64]);  add_2 = None
        permute_49: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_55, [1, 0, 2, 3]);  view_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_51: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_49, [0, 2, 1, 3]);  permute_49 = None
        view_57: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_51, [48, 1024, 64]);  permute_51 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_2: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_57, [0, 0, 256, 256], -1.0);  view_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_5: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_2, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_18: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
        permute_53: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_18, [0, 1, 4, 3, 2]);  unsqueeze_18 = None
        permute_55: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_53, [0, 1, 4, 3, 2]);  permute_53 = None
        clone_8: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_61: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_8, [192, 768, 64]);  clone_8 = None
        bmm_1: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_60, view_61);  view_60 = view_61 = None
        view_62: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_1, [48, 4, 256, 1, 64]);  bmm_1 = None
        permute_56: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_62, [0, 1, 2, 4, 3]);  view_62 = None
        view_63: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_56, [48, 4, 256, 64]);  permute_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_64: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_63, [4, 12, 1024, 64]);  view_63 = None
        permute_57: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_58: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_57, [1, 0, 2, 3]);  permute_57 = None
        clone_9: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_65: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_9, [1024, 4, 768]);  clone_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_59: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_65, [1, 0, 2]);  view_65 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_10: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_66: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_10, [4096, 768]);  clone_10 = None
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_3: "f32[4096, 768]" = torch.ops.aten.mm.default(view_66, permute_60);  view_66 = permute_60 = None
        view_67: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_3, [4, 1024, 768]);  mm_3 = None
        add_7: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_67, arg10_1);  view_67 = arg10_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_8: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_7, arg0_1);  add_7 = arg0_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 1024, 1]" = var_mean[0]
        getitem_1: "f32[4, 1024, 1]" = var_mean[1];  var_mean = None
        sub_6: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_1);  add_8 = getitem_1 = None
        add_9: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_1: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = rsqrt = None
        mul_2: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg11_1);  mul_1 = arg11_1 = None
        add_10: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, arg12_1);  mul_2 = arg12_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_68: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_10, [4096, 768])
        permute_61: "f32[768, 3072]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        
        # No stacktrace found for following nodes
        mm_default_23: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_68, permute_61);  view_68 = permute_61 = None
        add_tensor_23: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_23, arg14_1);  mm_default_23 = arg14_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_69: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_23, [4, 1024, 3072]);  add_tensor_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_3: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.5)
        mul_4: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476);  view_69 = None
        erf: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_11: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_3, add_11);  mul_3 = add_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_70: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_5, [4096, 3072]);  mul_5 = None
        permute_62: "f32[3072, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        
        # No stacktrace found for following nodes
        mm_default_22: "f32[4096, 768]" = torch.ops.aten.mm.default(view_70, permute_62);  view_70 = permute_62 = None
        add_tensor_22: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_22, arg16_1);  mm_default_22 = arg16_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_71: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_22, [4, 1024, 768]);  add_tensor_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_12: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_71, add_10);  view_71 = add_10 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_2: "f32[4, 1024, 1]" = var_mean_1[0]
        getitem_3: "f32[4, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_34: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_35: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 3);  unsqueeze_34 = None
        full_default_19: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_9: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_241: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1);  slice_241 = None
        slice_237: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_239: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_237, 3, 256, 9223372036854775807);  slice_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_7: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_3);  add_12 = getitem_3 = None
        add_13: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_6: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_1);  sub_7 = rsqrt_1 = None
        mul_7: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, arg17_1);  mul_6 = arg17_1 = None
        add_14: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_7, arg18_1);  mul_7 = arg18_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_63: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_14, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_13: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format)
        view_72: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_13, [4096, 768]);  clone_13 = None
        permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_4: "f32[4096, 768]" = torch.ops.aten.mm.default(view_72, permute_64);  view_72 = permute_64 = None
        view_73: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_4, [1024, 4, 768]);  mm_4 = None
        add_15: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_73, arg20_1);  view_73 = arg20_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_10: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_15, 8.0);  add_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_84: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_10, [1024, 4, 12, 64]);  div_10 = None
        permute_74: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_84, [1, 0, 2, 3]);  view_84 = None
        permute_75: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_74, [0, 2, 1, 3]);  permute_74 = None
        view_85: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_75, [48, 1024, 64]);  permute_75 = None
        view_86: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_85, [48, 2, 512, 64]);  view_85 = None
        as_strided_8: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_86, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_86 = None
        unsqueeze_21: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
        permute_76: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_21, [0, 1, 2, 4, 3]);  unsqueeze_21 = None
        permute_77: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_76, [0, 1, 2, 4, 3]);  permute_76 = None
        clone_16: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_87: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_16, [144, 512, 64]);  clone_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_14: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format)
        view_74: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_14, [4096, 768]);  clone_14 = None
        permute_65: "f32[768, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_5: "f32[4096, 768]" = torch.ops.aten.mm.default(view_74, permute_65);  view_74 = permute_65 = None
        view_75: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_5, [1024, 4, 768]);  mm_5 = None
        add_16: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_75, arg22_1);  view_75 = arg22_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_79: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_16, [1024, 4, 12, 64]);  add_16 = None
        permute_68: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_79, [1, 0, 2, 3]);  view_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_70: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_68, [0, 2, 1, 3]);  permute_68 = None
        view_81: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_70, [48, 1024, 64]);  permute_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_83: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_81, [48, 2, 512, 64]);  view_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_7: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_83, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_20: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
        permute_72: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_20, [0, 1, 4, 2, 3]);  unsqueeze_20 = None
        permute_78: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_72, [0, 1, 4, 3, 2]);  permute_72 = None
        clone_17: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_88: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_17, [144, 64, 512]);  clone_17 = None
        bmm_2: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_87, view_88);  view_87 = view_88 = None
        view_89: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_2, [48, 3, 512, 1, 512]);  bmm_2 = None
        permute_79: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_89, [0, 1, 2, 4, 3]);  view_89 = None
        view_90: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_79, [48, 3, 512, 512]);  permute_79 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_4: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_90, [0, 0, 0, 1], 0.0);  view_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_91: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_4, [48, 3, 512, 513]);  constant_pad_nd_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_234: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_91, 2, 0, 256)
        slice_235: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_234, 3, 0, 257);  slice_234 = None
        copy_12: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_239, slice_235);  slice_239 = slice_235 = None
        
        # No stacktrace found for following nodes
        slice_tensor_10: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_scatter_default_22: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_10, copy_12, 3, 256, 9223372036854775807);  slice_tensor_10 = copy_12 = None
        slice_scatter_default_23: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_default_22, 1, 0, -1);  full_9 = slice_scatter_default_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_23: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_23, 1, -1);  select_23 = None
        select_22: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_23, 1, -1)
        slice_255: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_22, 2, 256, 9223372036854775807);  select_22 = None
        select_20: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_91, 1, -1)
        slice_248: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_20, 1, 256, 9223372036854775807);  select_20 = None
        slice_249: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_248, 2, 0, 257);  slice_248 = None
        copy_13: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_255, slice_249);  slice_255 = slice_249 = None
        
        # No stacktrace found for following nodes
        select_int_4: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_23, 1, -1)
        slice_scatter_default_24: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_4, copy_13, 2, 256, 9223372036854775807);  select_int_4 = copy_13 = None
        select_scatter_default_4: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_23, slice_scatter_default_24, 1, -1);  slice_scatter_default_23 = slice_scatter_default_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_274: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_4, 1, 1, 9223372036854775807);  slice_274 = None
        slice_270: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_4, 1, 1, 9223372036854775807)
        slice_272: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_270, 3, 0, 256);  slice_270 = None
        slice_263: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_91, 2, -257, -1)
        slice_264: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_263, 3, 257, 9223372036854775807);  slice_263 = None
        copy_14: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_272, slice_264);  slice_272 = slice_264 = None
        
        # No stacktrace found for following nodes
        slice_tensor_11: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_4, 1, 1, 9223372036854775807)
        slice_scatter_default_25: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_11, copy_14, 3, 0, 256);  slice_tensor_11 = copy_14 = None
        slice_scatter_default_26: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_4, slice_scatter_default_25, 1, 1, 9223372036854775807);  select_scatter_default_4 = slice_scatter_default_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_28: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_26, 1, 0)
        slice_290: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_28, 1, 1, 256);  select_28 = slice_290 = None
        select_27: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_26, 1, 0)
        slice_287: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_27, 1, 1, 256);  select_27 = None
        slice_288: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_287, 2, 1, 256);  slice_287 = None
        select_25: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_91, 1, 0);  view_91 = None
        slice_281: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_25, 1, 0, 255);  select_25 = None
        slice_282: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_281, 2, -255, 9223372036854775807);  slice_281 = None
        copy_15: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_288, slice_282);  slice_288 = slice_282 = None
        
        # No stacktrace found for following nodes
        select_int_5: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_26, 1, 0)
        slice_tensor_12: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_5, 1, 1, 256)
        slice_scatter_default_27: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_12, copy_15, 2, 1, 256);  slice_tensor_12 = copy_15 = None
        slice_scatter_default_28: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_5, slice_scatter_default_27, 1, 1, 256);  select_int_5 = slice_scatter_default_27 = None
        select_scatter_default_5: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_26, slice_scatter_default_28, 1, 0);  slice_scatter_default_26 = slice_scatter_default_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_96: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_5, [4, 12, 1024, 513])
        permute_84: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        slice_313: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_84, 1, 0, 256);  slice_313 = None
        view_95: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_5, [4, 12, 1024, 513])
        permute_83: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        slice_309: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_83, 1, 0, 256);  permute_83 = None
        slice_311: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_309, 3, 0, 257);  slice_309 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_4: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_22: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_4, -2);  iota_4 = None
        iota_5: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_23: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        sub_9: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_22, unsqueeze_23);  unsqueeze_22 = unsqueeze_23 = None
        le_2: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_9, 0);  sub_9 = None
        full_default_10: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[256, 257]" = torch.ops.aten.where.self(le_2, full_default_10, full_default_11);  le_2 = full_default_10 = full_default_11 = None
        rev_4: "f32[256, 257]" = torch.ops.prims.rev.default(where_8, [0]);  where_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_24: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_4, 0);  rev_4 = None
        unsqueeze_25: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_4: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_25, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_5: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_4, torch.bool);  expand_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_12: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_94: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_5, [4, 12, 1024, 513]);  select_scatter_default_5 = None
        permute_82: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        slice_301: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_82, 1, 0, 256);  permute_82 = None
        slice_303: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_301, 3, 0, 257);  slice_301 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_9: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_5, full_default_12, slice_303);  convert_element_type_5 = full_default_12 = slice_303 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_16: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_311, where_9);  slice_311 = where_9 = None
        
        # No stacktrace found for following nodes
        slice_tensor_13: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_84, 1, 0, 256)
        slice_scatter_default_29: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_13, copy_16, 3, 0, 257);  slice_tensor_13 = copy_16 = None
        slice_scatter_default_30: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_84, slice_scatter_default_29, 1, 0, 256);  permute_84 = slice_scatter_default_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_85: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_30, [0, 2, 1, 3]);  slice_scatter_default_30 = None
        view_97: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_85, [48, 4, 256, 513]);  permute_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_101: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_97, [4, 12, 1024, 513])
        permute_89: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        slice_336: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_89, 1, -256, 9223372036854775807);  slice_336 = None
        view_100: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_97, [4, 12, 1024, 513])
        permute_88: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        slice_332: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_88, 1, -256, 9223372036854775807);  permute_88 = None
        slice_334: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_332, 3, -257, 9223372036854775807);  slice_332 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_5: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_25, [1, 3]);  unsqueeze_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_5: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_5, [4, 256, 12, 257]);  rev_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_6: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_5, torch.bool);  expand_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_13: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_99: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_97, [4, 12, 1024, 513]);  view_97 = None
        permute_87: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        slice_324: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_87, 1, -256, 9223372036854775807);  permute_87 = None
        slice_326: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_324, 3, -257, 9223372036854775807);  slice_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_10: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_6, full_default_13, slice_326);  convert_element_type_6 = full_default_13 = slice_326 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_17: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_334, where_10);  slice_334 = where_10 = None
        
        # No stacktrace found for following nodes
        slice_tensor_14: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_89, 1, -256, 9223372036854775807)
        slice_scatter_default_31: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_14, copy_17, 3, -257, 9223372036854775807);  slice_tensor_14 = copy_17 = None
        slice_scatter_default_32: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_89, slice_scatter_default_31, 1, -256, 9223372036854775807);  permute_89 = slice_scatter_default_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_90: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_32, [0, 2, 1, 3]);  slice_scatter_default_32 = None
        view_102: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_90, [48, 4, 256, 513]);  permute_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_122: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_102, [4, 12, 1024, 513]);  view_102 = None
        permute_108: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_14: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_353: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1);  slice_353 = None
        slice_349: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1)
        slice_351: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_349, 3, 256, 9223372036854775807);  slice_349 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_13: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_92: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_13, [0, 2, 1, 3]);  full_13 = None
        view_104: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_92, [4, 1024, 1]);  permute_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_106: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_104, [4, 2, 512, 1]);  view_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_9: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_106, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_28: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_9, 4);  as_strided_9 = None
        permute_94: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_28, [0, 1, 2, 4, 3]);  unsqueeze_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_14: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_1: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_26: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_1, 2);  ne_1 = None
        unsqueeze_27: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 3);  unsqueeze_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_7: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_27, torch.float32)
        where_11: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_27, full_default_14, convert_element_type_7);  unsqueeze_27 = full_default_14 = convert_element_type_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_93: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_11, [0, 2, 1, 3]);  where_11 = None
        view_105: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_93, [4, 1024, 1]);  permute_93 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_107: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_105, [4, 2, 512, 1]);  view_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_10: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_107, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_29: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_10, 4);  as_strided_10 = None
        permute_95: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_29, [0, 1, 4, 2, 3]);  unsqueeze_29 = None
        mul_8: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_94, permute_95);  permute_94 = permute_95 = None
        view_108: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_8, [4, 3, 512, 512]);  mul_8 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_5: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_108, [0, 0, 0, 1], 0.0);  view_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_109: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_5, [4, 3, 512, 513]);  constant_pad_nd_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_346: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_109, 2, 0, 256)
        slice_347: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_346, 3, 0, 257);  slice_346 = None
        copy_18: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_351, slice_347);  slice_351 = slice_347 = None
        
        # No stacktrace found for following nodes
        slice_tensor_15: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_14, 1, 0, -1)
        slice_scatter_default_33: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_15, copy_18, 3, 256, 9223372036854775807);  slice_tensor_15 = copy_18 = None
        slice_scatter_default_34: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_14, slice_scatter_default_33, 1, 0, -1);  full_14 = slice_scatter_default_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_33: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_34, 1, -1);  select_33 = None
        select_32: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_34, 1, -1)
        slice_367: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_32, 2, 256, 9223372036854775807);  select_32 = None
        select_30: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_109, 1, -1)
        slice_360: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_30, 1, 256, 9223372036854775807);  select_30 = None
        slice_361: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_360, 2, 0, 257);  slice_360 = None
        copy_19: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_367, slice_361);  slice_367 = slice_361 = None
        
        # No stacktrace found for following nodes
        select_int_6: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_34, 1, -1)
        slice_scatter_default_35: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_6, copy_19, 2, 256, 9223372036854775807);  select_int_6 = copy_19 = None
        select_scatter_default_6: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_34, slice_scatter_default_35, 1, -1);  slice_scatter_default_34 = slice_scatter_default_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_386: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_6, 1, 1, 9223372036854775807);  slice_386 = None
        slice_382: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_6, 1, 1, 9223372036854775807)
        slice_384: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_382, 3, 0, 256);  slice_382 = None
        slice_375: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_109, 2, -257, -1)
        slice_376: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_375, 3, 257, 9223372036854775807);  slice_375 = None
        copy_20: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_384, slice_376);  slice_384 = slice_376 = None
        
        # No stacktrace found for following nodes
        slice_tensor_16: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_6, 1, 1, 9223372036854775807)
        slice_scatter_default_36: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_16, copy_20, 3, 0, 256);  slice_tensor_16 = copy_20 = None
        slice_scatter_default_37: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_6, slice_scatter_default_36, 1, 1, 9223372036854775807);  select_scatter_default_6 = slice_scatter_default_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_38: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_37, 1, 0)
        slice_402: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_38, 1, 1, 256);  select_38 = slice_402 = None
        select_37: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_37, 1, 0)
        slice_399: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_37, 1, 1, 256);  select_37 = None
        slice_400: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_399, 2, 1, 256);  slice_399 = None
        select_35: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_109, 1, 0);  view_109 = None
        slice_393: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_35, 1, 0, 255);  select_35 = None
        slice_394: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_393, 2, -255, 9223372036854775807);  slice_393 = None
        copy_21: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_400, slice_394);  slice_400 = slice_394 = None
        
        # No stacktrace found for following nodes
        select_int_7: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_37, 1, 0)
        slice_tensor_17: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_7, 1, 1, 256)
        slice_scatter_default_38: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_17, copy_21, 2, 1, 256);  slice_tensor_17 = copy_21 = None
        slice_scatter_default_39: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_7, slice_scatter_default_38, 1, 1, 256);  select_int_7 = slice_scatter_default_38 = None
        select_scatter_default_7: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_37, slice_scatter_default_39, 1, 0);  slice_scatter_default_37 = slice_scatter_default_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_114: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_7, [4, 1, 1024, 513])
        permute_100: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        slice_425: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_100, 1, 0, 256);  slice_425 = None
        view_113: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_7, [4, 1, 1024, 513])
        permute_99: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        slice_421: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_99, 1, 0, 256);  permute_99 = None
        slice_423: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_421, 3, 0, 257);  slice_421 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_6: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_30: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_6, -2);  iota_6 = None
        iota_7: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_31: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        sub_11: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_30, unsqueeze_31);  unsqueeze_30 = unsqueeze_31 = None
        le_3: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_11, 0);  sub_11 = None
        full_default_15: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f32[256, 257]" = torch.ops.aten.where.self(le_3, full_default_15, full_default_16);  le_3 = full_default_15 = full_default_16 = None
        rev_6: "f32[256, 257]" = torch.ops.prims.rev.default(where_12, [0]);  where_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_32: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_6, 0);  rev_6 = None
        unsqueeze_33: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 2);  unsqueeze_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_6: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_33, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_8: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_6, torch.bool);  expand_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_17: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_112: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_7, [4, 1, 1024, 513]);  select_scatter_default_7 = None
        permute_98: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        slice_413: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_98, 1, 0, 256);  permute_98 = None
        slice_415: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_413, 3, 0, 257);  slice_413 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_13: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_8, full_default_17, slice_415);  convert_element_type_8 = full_default_17 = slice_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_22: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_423, where_13);  slice_423 = where_13 = None
        
        # No stacktrace found for following nodes
        slice_tensor_18: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_100, 1, 0, 256)
        slice_scatter_default_40: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_18, copy_22, 3, 0, 257);  slice_tensor_18 = copy_22 = None
        slice_scatter_default_41: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_100, slice_scatter_default_40, 1, 0, 256);  permute_100 = slice_scatter_default_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_101: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_41, [0, 2, 1, 3]);  slice_scatter_default_41 = None
        view_115: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_101, [4, 4, 256, 513]);  permute_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_119: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_115, [4, 1, 1024, 513])
        permute_105: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        slice_448: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_105, 1, -256, 9223372036854775807);  slice_448 = None
        view_118: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_115, [4, 1, 1024, 513])
        permute_104: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        slice_444: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_104, 1, -256, 9223372036854775807);  permute_104 = None
        slice_446: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_444, 3, -257, 9223372036854775807);  slice_444 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_7: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_33, [1, 3]);  unsqueeze_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_7: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_7, [4, 256, 1, 257]);  rev_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_9: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_7, torch.bool);  expand_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_18: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_117: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_115, [4, 1, 1024, 513]);  view_115 = None
        permute_103: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        slice_436: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_103, 1, -256, 9223372036854775807);  permute_103 = None
        slice_438: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_436, 3, -257, 9223372036854775807);  slice_436 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_14: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_9, full_default_18, slice_438);  convert_element_type_9 = full_default_18 = slice_438 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_23: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_446, where_14);  slice_446 = where_14 = None
        
        # No stacktrace found for following nodes
        slice_tensor_19: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_105, 1, -256, 9223372036854775807)
        slice_scatter_default_42: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_19, copy_23, 3, -257, 9223372036854775807);  slice_tensor_19 = copy_23 = None
        slice_scatter_default_43: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_105, slice_scatter_default_42, 1, -256, 9223372036854775807);  permute_105 = slice_scatter_default_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_106: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_43, [0, 2, 1, 3]);  slice_scatter_default_43 = None
        view_120: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_106, [4, 4, 256, 513]);  permute_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_123: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_120, [4, 1, 1024, 513]);  view_120 = None
        permute_109: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        add_20: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_108, permute_109);  permute_108 = permute_109 = None
        permute_110: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_20, [0, 2, 1, 3]);  add_20 = None
        view_125: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_110, [48, 4, 256, 513]);  permute_110 = None
        view_126: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_125, [4, 12, 1024, 513]);  view_125 = None
        permute_111: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_18: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        amax_1: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_18, [-1], True)
        sub_12: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_18, amax_1);  clone_18 = amax_1 = None
        exp_1: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_2: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_17: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_15: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_35, full_default_19, div_17);  unsqueeze_35 = full_default_19 = div_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_113: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_15, [0, 2, 1, 3]);  where_15 = None
        clone_20: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_128: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_20, [48, 4, 256, 513]);  clone_20 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_128, [0, 257], 0.0);  view_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_130: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_7, [48, 4, -1]);  constant_pad_nd_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_458: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_130, 2, 0, -256);  view_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_131: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_458, [48, 4, 256, 769]);  slice_458 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_462: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_131, 3, 0, -1);  view_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_36: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_462, 4);  slice_462 = None
        permute_115: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_36, [0, 1, 2, 4, 3]);  unsqueeze_36 = None
        permute_117: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_115, [0, 1, 2, 4, 3]);  permute_115 = None
        view_132: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_117, [192, 256, 768]);  permute_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_15: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        view_76: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_15, [4096, 768]);  clone_15 = None
        permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_6: "f32[4096, 768]" = torch.ops.aten.mm.default(view_76, permute_66);  view_76 = permute_66 = None
        view_77: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_6, [1024, 4, 768]);  mm_6 = None
        add_17: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_77, arg24_1);  view_77 = arg24_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_127: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_17, [1024, 4, 12, 64]);  add_17 = None
        permute_112: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_127, [1, 0, 2, 3]);  view_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_114: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_112, [0, 2, 1, 3]);  permute_112 = None
        view_129: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_114, [48, 1024, 64]);  permute_114 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_129, [0, 0, 256, 256], -1.0);  view_129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_11: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_6, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_37: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_11, 4);  as_strided_11 = None
        permute_116: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_37, [0, 1, 4, 3, 2]);  unsqueeze_37 = None
        permute_118: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_116, [0, 1, 4, 3, 2]);  permute_116 = None
        clone_21: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
        view_133: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_21, [192, 768, 64]);  clone_21 = None
        bmm_3: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_132, view_133);  view_132 = view_133 = None
        view_134: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_3, [48, 4, 256, 1, 64]);  bmm_3 = None
        permute_119: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_134, [0, 1, 2, 4, 3]);  view_134 = None
        view_135: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_119, [48, 4, 256, 64]);  permute_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_136: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_135, [4, 12, 1024, 64]);  view_135 = None
        permute_120: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_121: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_120, [1, 0, 2, 3]);  permute_120 = None
        clone_22: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_137: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_22, [1024, 4, 768]);  clone_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_122: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_137, [1, 0, 2]);  view_137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_23: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_138: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_23, [4096, 768]);  clone_23 = None
        permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_7: "f32[4096, 768]" = torch.ops.aten.mm.default(view_138, permute_123);  view_138 = permute_123 = None
        view_139: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_7, [4, 1024, 768]);  mm_7 = None
        add_22: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_139, arg26_1);  view_139 = arg26_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_23: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_22, add_14);  add_22 = add_14 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
        getitem_4: "f32[4, 1024, 1]" = var_mean_2[0]
        getitem_5: "f32[4, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_14: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_5);  add_23 = getitem_5 = None
        add_24: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_9: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_2);  sub_14 = rsqrt_2 = None
        mul_10: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg27_1);  mul_9 = arg27_1 = None
        add_25: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_10, arg28_1);  mul_10 = arg28_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_140: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_25, [4096, 768])
        permute_124: "f32[768, 3072]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        
        # No stacktrace found for following nodes
        mm_default_21: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_140, permute_124);  view_140 = permute_124 = None
        add_tensor_21: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_21, arg30_1);  mm_default_21 = arg30_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_141: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_21, [4, 1024, 3072]);  add_tensor_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_11: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_12: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_1: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_26: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_26);  mul_11 = add_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_142: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_13, [4096, 3072]);  mul_13 = None
        permute_125: "f32[3072, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        
        # No stacktrace found for following nodes
        mm_default_20: "f32[4096, 768]" = torch.ops.aten.mm.default(view_142, permute_125);  view_142 = permute_125 = None
        add_tensor_20: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_20, arg32_1);  mm_default_20 = arg32_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_143: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [4, 1024, 768]);  add_tensor_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_27: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_143, add_25);  view_143 = add_25 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_6: "f32[4, 1024, 1]" = var_mean_3[0]
        getitem_7: "f32[4, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_53: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_54: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 3);  unsqueeze_53 = None
        full_default_29: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_18: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_472: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1);  slice_472 = None
        slice_468: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_470: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_468, 3, 256, 9223372036854775807);  slice_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_15: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_7);  add_27 = getitem_7 = None
        add_28: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_14: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_3);  sub_15 = rsqrt_3 = None
        mul_15: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        add_29: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_15, arg34_1);  mul_15 = arg34_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_126: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_29, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_26: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format)
        view_144: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_26, [4096, 768]);  clone_26 = None
        permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_8: "f32[4096, 768]" = torch.ops.aten.mm.default(view_144, permute_127);  view_144 = permute_127 = None
        view_145: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_8, [1024, 4, 768]);  mm_8 = None
        add_30: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_145, arg36_1);  view_145 = arg36_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_20: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_30, 8.0);  add_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_156: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_20, [1024, 4, 12, 64]);  div_20 = None
        permute_137: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_156, [1, 0, 2, 3]);  view_156 = None
        permute_138: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_137, [0, 2, 1, 3]);  permute_137 = None
        view_157: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_138, [48, 1024, 64]);  permute_138 = None
        view_158: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_157, [48, 2, 512, 64]);  view_157 = None
        as_strided_14: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_158, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_158 = None
        unsqueeze_40: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
        permute_139: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_40, [0, 1, 2, 4, 3]);  unsqueeze_40 = None
        permute_140: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_139, [0, 1, 2, 4, 3]);  permute_139 = None
        clone_29: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_159: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_29, [144, 512, 64]);  clone_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_27: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format)
        view_146: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_27, [4096, 768]);  clone_27 = None
        permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_9: "f32[4096, 768]" = torch.ops.aten.mm.default(view_146, permute_128);  view_146 = permute_128 = None
        view_147: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_9, [1024, 4, 768]);  mm_9 = None
        add_31: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_147, arg38_1);  view_147 = arg38_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_151: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_31, [1024, 4, 12, 64]);  add_31 = None
        permute_131: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_151, [1, 0, 2, 3]);  view_151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_133: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_131, [0, 2, 1, 3]);  permute_131 = None
        view_153: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_133, [48, 1024, 64]);  permute_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_155: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_153, [48, 2, 512, 64]);  view_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_13: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_155, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_39: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_13, 4);  as_strided_13 = None
        permute_135: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_39, [0, 1, 4, 2, 3]);  unsqueeze_39 = None
        permute_141: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_135, [0, 1, 4, 3, 2]);  permute_135 = None
        clone_30: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        view_160: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_30, [144, 64, 512]);  clone_30 = None
        bmm_4: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_159, view_160);  view_159 = view_160 = None
        view_161: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_4, [48, 3, 512, 1, 512]);  bmm_4 = None
        permute_142: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_161, [0, 1, 2, 4, 3]);  view_161 = None
        view_162: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_142, [48, 3, 512, 512]);  permute_142 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_162, [0, 0, 0, 1], 0.0);  view_162 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_163: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_8, [48, 3, 512, 513]);  constant_pad_nd_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_465: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_163, 2, 0, 256)
        slice_466: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_465, 3, 0, 257);  slice_465 = None
        copy_24: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_470, slice_466);  slice_470 = slice_466 = None
        
        # No stacktrace found for following nodes
        slice_tensor_20: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_scatter_default_44: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_20, copy_24, 3, 256, 9223372036854775807);  slice_tensor_20 = copy_24 = None
        slice_scatter_default_45: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_18, slice_scatter_default_44, 1, 0, -1);  full_18 = slice_scatter_default_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_43: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_45, 1, -1);  select_43 = None
        select_42: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_45, 1, -1)
        slice_486: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_42, 2, 256, 9223372036854775807);  select_42 = None
        select_40: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_163, 1, -1)
        slice_479: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_40, 1, 256, 9223372036854775807);  select_40 = None
        slice_480: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_479, 2, 0, 257);  slice_479 = None
        copy_25: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_486, slice_480);  slice_486 = slice_480 = None
        
        # No stacktrace found for following nodes
        select_int_8: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_45, 1, -1)
        slice_scatter_default_46: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_8, copy_25, 2, 256, 9223372036854775807);  select_int_8 = copy_25 = None
        select_scatter_default_8: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_45, slice_scatter_default_46, 1, -1);  slice_scatter_default_45 = slice_scatter_default_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_505: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_8, 1, 1, 9223372036854775807);  slice_505 = None
        slice_501: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_8, 1, 1, 9223372036854775807)
        slice_503: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_501, 3, 0, 256);  slice_501 = None
        slice_494: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_163, 2, -257, -1)
        slice_495: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_494, 3, 257, 9223372036854775807);  slice_494 = None
        copy_26: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_503, slice_495);  slice_503 = slice_495 = None
        
        # No stacktrace found for following nodes
        slice_tensor_21: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_8, 1, 1, 9223372036854775807)
        slice_scatter_default_47: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_21, copy_26, 3, 0, 256);  slice_tensor_21 = copy_26 = None
        slice_scatter_default_48: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_8, slice_scatter_default_47, 1, 1, 9223372036854775807);  select_scatter_default_8 = slice_scatter_default_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_48: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_48, 1, 0)
        slice_521: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_48, 1, 1, 256);  select_48 = slice_521 = None
        select_47: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_48, 1, 0)
        slice_518: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_47, 1, 1, 256);  select_47 = None
        slice_519: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_518, 2, 1, 256);  slice_518 = None
        select_45: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_163, 1, 0);  view_163 = None
        slice_512: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_45, 1, 0, 255);  select_45 = None
        slice_513: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_512, 2, -255, 9223372036854775807);  slice_512 = None
        copy_27: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_519, slice_513);  slice_519 = slice_513 = None
        
        # No stacktrace found for following nodes
        select_int_9: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_48, 1, 0)
        slice_tensor_22: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_9, 1, 1, 256)
        slice_scatter_default_49: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_22, copy_27, 2, 1, 256);  slice_tensor_22 = copy_27 = None
        slice_scatter_default_50: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_9, slice_scatter_default_49, 1, 1, 256);  select_int_9 = slice_scatter_default_49 = None
        select_scatter_default_9: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_48, slice_scatter_default_50, 1, 0);  slice_scatter_default_48 = slice_scatter_default_50 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_168: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_9, [4, 12, 1024, 513])
        permute_147: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        slice_544: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_147, 1, 0, 256);  slice_544 = None
        view_167: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_9, [4, 12, 1024, 513])
        permute_146: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        slice_540: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_146, 1, 0, 256);  permute_146 = None
        slice_542: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_540, 3, 0, 257);  slice_540 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_8: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_41: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_8, -2);  iota_8 = None
        iota_9: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_42: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
        sub_17: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_41, unsqueeze_42);  unsqueeze_41 = unsqueeze_42 = None
        le_4: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_17, 0);  sub_17 = None
        full_default_20: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f32[256, 257]" = torch.ops.aten.where.self(le_4, full_default_20, full_default_21);  le_4 = full_default_20 = full_default_21 = None
        rev_8: "f32[256, 257]" = torch.ops.prims.rev.default(where_16, [0]);  where_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_43: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_8, 0);  rev_8 = None
        unsqueeze_44: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 2);  unsqueeze_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_8: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_44, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_10: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_8, torch.bool);  expand_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_22: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_166: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_9, [4, 12, 1024, 513]);  select_scatter_default_9 = None
        permute_145: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        slice_532: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_145, 1, 0, 256);  permute_145 = None
        slice_534: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_532, 3, 0, 257);  slice_532 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_17: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_10, full_default_22, slice_534);  convert_element_type_10 = full_default_22 = slice_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_28: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_542, where_17);  slice_542 = where_17 = None
        
        # No stacktrace found for following nodes
        slice_tensor_23: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_147, 1, 0, 256)
        slice_scatter_default_51: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_23, copy_28, 3, 0, 257);  slice_tensor_23 = copy_28 = None
        slice_scatter_default_52: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_147, slice_scatter_default_51, 1, 0, 256);  permute_147 = slice_scatter_default_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_148: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_52, [0, 2, 1, 3]);  slice_scatter_default_52 = None
        view_169: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_148, [48, 4, 256, 513]);  permute_148 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_173: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_169, [4, 12, 1024, 513])
        permute_152: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        slice_567: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_152, 1, -256, 9223372036854775807);  slice_567 = None
        view_172: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_169, [4, 12, 1024, 513])
        permute_151: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        slice_563: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_151, 1, -256, 9223372036854775807);  permute_151 = None
        slice_565: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_563, 3, -257, 9223372036854775807);  slice_563 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_9: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_44, [1, 3]);  unsqueeze_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_9: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_9, [4, 256, 12, 257]);  rev_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_11: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_9, torch.bool);  expand_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_23: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_171: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_169, [4, 12, 1024, 513]);  view_169 = None
        permute_150: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        slice_555: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_150, 1, -256, 9223372036854775807);  permute_150 = None
        slice_557: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_555, 3, -257, 9223372036854775807);  slice_555 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_18: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_11, full_default_23, slice_557);  convert_element_type_11 = full_default_23 = slice_557 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_29: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_565, where_18);  slice_565 = where_18 = None
        
        # No stacktrace found for following nodes
        slice_tensor_24: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_152, 1, -256, 9223372036854775807)
        slice_scatter_default_53: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_24, copy_29, 3, -257, 9223372036854775807);  slice_tensor_24 = copy_29 = None
        slice_scatter_default_54: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_152, slice_scatter_default_53, 1, -256, 9223372036854775807);  permute_152 = slice_scatter_default_53 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_153: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_54, [0, 2, 1, 3]);  slice_scatter_default_54 = None
        view_174: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_153, [48, 4, 256, 513]);  permute_153 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_194: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_174, [4, 12, 1024, 513]);  view_174 = None
        permute_171: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_23: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_584: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1);  slice_584 = None
        slice_580: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_582: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_580, 3, 256, 9223372036854775807);  slice_580 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_22: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_155: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_22, [0, 2, 1, 3]);  full_22 = None
        view_176: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_155, [4, 1024, 1]);  permute_155 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_178: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_176, [4, 2, 512, 1]);  view_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_15: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_178, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_47: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_15, 4);  as_strided_15 = None
        permute_157: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_47, [0, 1, 2, 4, 3]);  unsqueeze_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_24: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_2: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_45: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_2, 2);  ne_2 = None
        unsqueeze_46: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 3);  unsqueeze_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_12: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_46, torch.float32)
        where_19: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_46, full_default_24, convert_element_type_12);  unsqueeze_46 = full_default_24 = convert_element_type_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_156: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_19, [0, 2, 1, 3]);  where_19 = None
        view_177: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_156, [4, 1024, 1]);  permute_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_179: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_177, [4, 2, 512, 1]);  view_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_16: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_179, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_48: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_16, 4);  as_strided_16 = None
        permute_158: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_48, [0, 1, 4, 2, 3]);  unsqueeze_48 = None
        mul_16: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_157, permute_158);  permute_157 = permute_158 = None
        view_180: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_16, [4, 3, 512, 512]);  mul_16 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_9: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_180, [0, 0, 0, 1], 0.0);  view_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_181: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_9, [4, 3, 512, 513]);  constant_pad_nd_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_577: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_181, 2, 0, 256)
        slice_578: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_577, 3, 0, 257);  slice_577 = None
        copy_30: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_582, slice_578);  slice_582 = slice_578 = None
        
        # No stacktrace found for following nodes
        slice_tensor_25: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_scatter_default_55: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_25, copy_30, 3, 256, 9223372036854775807);  slice_tensor_25 = copy_30 = None
        slice_scatter_default_56: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_default_55, 1, 0, -1);  full_23 = slice_scatter_default_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_53: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_56, 1, -1);  select_53 = None
        select_52: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_56, 1, -1)
        slice_598: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_52, 2, 256, 9223372036854775807);  select_52 = None
        select_50: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_181, 1, -1)
        slice_591: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_50, 1, 256, 9223372036854775807);  select_50 = None
        slice_592: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_591, 2, 0, 257);  slice_591 = None
        copy_31: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_598, slice_592);  slice_598 = slice_592 = None
        
        # No stacktrace found for following nodes
        select_int_10: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_56, 1, -1)
        slice_scatter_default_57: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_10, copy_31, 2, 256, 9223372036854775807);  select_int_10 = copy_31 = None
        select_scatter_default_10: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_56, slice_scatter_default_57, 1, -1);  slice_scatter_default_56 = slice_scatter_default_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_617: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_10, 1, 1, 9223372036854775807);  slice_617 = None
        slice_613: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_10, 1, 1, 9223372036854775807)
        slice_615: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_613, 3, 0, 256);  slice_613 = None
        slice_606: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_181, 2, -257, -1)
        slice_607: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_606, 3, 257, 9223372036854775807);  slice_606 = None
        copy_32: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_615, slice_607);  slice_615 = slice_607 = None
        
        # No stacktrace found for following nodes
        slice_tensor_26: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_10, 1, 1, 9223372036854775807)
        slice_scatter_default_58: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_26, copy_32, 3, 0, 256);  slice_tensor_26 = copy_32 = None
        slice_scatter_default_59: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_10, slice_scatter_default_58, 1, 1, 9223372036854775807);  select_scatter_default_10 = slice_scatter_default_58 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_58: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_59, 1, 0)
        slice_633: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_58, 1, 1, 256);  select_58 = slice_633 = None
        select_57: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_59, 1, 0)
        slice_630: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_57, 1, 1, 256);  select_57 = None
        slice_631: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_630, 2, 1, 256);  slice_630 = None
        select_55: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_181, 1, 0);  view_181 = None
        slice_624: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_55, 1, 0, 255);  select_55 = None
        slice_625: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_624, 2, -255, 9223372036854775807);  slice_624 = None
        copy_33: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_631, slice_625);  slice_631 = slice_625 = None
        
        # No stacktrace found for following nodes
        select_int_11: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_59, 1, 0)
        slice_tensor_27: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_11, 1, 1, 256)
        slice_scatter_default_60: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_27, copy_33, 2, 1, 256);  slice_tensor_27 = copy_33 = None
        slice_scatter_default_61: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_11, slice_scatter_default_60, 1, 1, 256);  select_int_11 = slice_scatter_default_60 = None
        select_scatter_default_11: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_59, slice_scatter_default_61, 1, 0);  slice_scatter_default_59 = slice_scatter_default_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_186: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_11, [4, 1, 1024, 513])
        permute_163: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        slice_656: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_163, 1, 0, 256);  slice_656 = None
        view_185: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_11, [4, 1, 1024, 513])
        permute_162: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        slice_652: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_162, 1, 0, 256);  permute_162 = None
        slice_654: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_652, 3, 0, 257);  slice_652 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_10: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_49: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_10, -2);  iota_10 = None
        iota_11: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_50: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        sub_19: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_49, unsqueeze_50);  unsqueeze_49 = unsqueeze_50 = None
        le_5: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_19, 0);  sub_19 = None
        full_default_25: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f32[256, 257]" = torch.ops.aten.where.self(le_5, full_default_25, full_default_26);  le_5 = full_default_25 = full_default_26 = None
        rev_10: "f32[256, 257]" = torch.ops.prims.rev.default(where_20, [0]);  where_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_51: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_10, 0);  rev_10 = None
        unsqueeze_52: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 2);  unsqueeze_51 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_10: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_52, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_13: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_10, torch.bool);  expand_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_27: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_184: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_11, [4, 1, 1024, 513]);  select_scatter_default_11 = None
        permute_161: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        slice_644: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_161, 1, 0, 256);  permute_161 = None
        slice_646: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_644, 3, 0, 257);  slice_644 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_21: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_13, full_default_27, slice_646);  convert_element_type_13 = full_default_27 = slice_646 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_34: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_654, where_21);  slice_654 = where_21 = None
        
        # No stacktrace found for following nodes
        slice_tensor_28: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_163, 1, 0, 256)
        slice_scatter_default_62: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_28, copy_34, 3, 0, 257);  slice_tensor_28 = copy_34 = None
        slice_scatter_default_63: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_163, slice_scatter_default_62, 1, 0, 256);  permute_163 = slice_scatter_default_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_164: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_63, [0, 2, 1, 3]);  slice_scatter_default_63 = None
        view_187: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_164, [4, 4, 256, 513]);  permute_164 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_191: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_187, [4, 1, 1024, 513])
        permute_168: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        slice_679: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_168, 1, -256, 9223372036854775807);  slice_679 = None
        view_190: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_187, [4, 1, 1024, 513])
        permute_167: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        slice_675: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_167, 1, -256, 9223372036854775807);  permute_167 = None
        slice_677: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_675, 3, -257, 9223372036854775807);  slice_675 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_11: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_52, [1, 3]);  unsqueeze_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_11: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_11, [4, 256, 1, 257]);  rev_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_14: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_11, torch.bool);  expand_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_28: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_189: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_187, [4, 1, 1024, 513]);  view_187 = None
        permute_166: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        slice_667: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_166, 1, -256, 9223372036854775807);  permute_166 = None
        slice_669: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_667, 3, -257, 9223372036854775807);  slice_667 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_22: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_14, full_default_28, slice_669);  convert_element_type_14 = full_default_28 = slice_669 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_35: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_677, where_22);  slice_677 = where_22 = None
        
        # No stacktrace found for following nodes
        slice_tensor_29: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_168, 1, -256, 9223372036854775807)
        slice_scatter_default_64: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_29, copy_35, 3, -257, 9223372036854775807);  slice_tensor_29 = copy_35 = None
        slice_scatter_default_65: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_168, slice_scatter_default_64, 1, -256, 9223372036854775807);  permute_168 = slice_scatter_default_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_169: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_65, [0, 2, 1, 3]);  slice_scatter_default_65 = None
        view_192: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_169, [4, 4, 256, 513]);  permute_169 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_195: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_192, [4, 1, 1024, 513]);  view_192 = None
        permute_172: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        add_35: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_171, permute_172);  permute_171 = permute_172 = None
        permute_173: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_35, [0, 2, 1, 3]);  add_35 = None
        view_197: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_173, [48, 4, 256, 513]);  permute_173 = None
        view_198: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_197, [4, 12, 1024, 513]);  view_197 = None
        permute_174: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_31: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
        amax_2: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_31, [-1], True)
        sub_20: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_31, amax_2);  clone_31 = amax_2 = None
        exp_2: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_3: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_27: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_23: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_54, full_default_29, div_27);  unsqueeze_54 = full_default_29 = div_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_176: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_23, [0, 2, 1, 3]);  where_23 = None
        clone_33: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_200: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_33, [48, 4, 256, 513]);  clone_33 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_11: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_200, [0, 257], 0.0);  view_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_202: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_11, [48, 4, -1]);  constant_pad_nd_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_689: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_202, 2, 0, -256);  view_202 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_203: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_689, [48, 4, 256, 769]);  slice_689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_693: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_203, 3, 0, -1);  view_203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_55: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_693, 4);  slice_693 = None
        permute_178: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_55, [0, 1, 2, 4, 3]);  unsqueeze_55 = None
        permute_180: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_178, [0, 1, 2, 4, 3]);  permute_178 = None
        view_204: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_180, [192, 256, 768]);  permute_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_28: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_148: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_28, [4096, 768]);  clone_28 = None
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_10: "f32[4096, 768]" = torch.ops.aten.mm.default(view_148, permute_129);  view_148 = permute_129 = None
        view_149: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_10, [1024, 4, 768]);  mm_10 = None
        add_32: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_149, arg40_1);  view_149 = arg40_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_199: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_32, [1024, 4, 12, 64]);  add_32 = None
        permute_175: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_199, [1, 0, 2, 3]);  view_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_177: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_175, [0, 2, 1, 3]);  permute_175 = None
        view_201: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_177, [48, 1024, 64]);  permute_177 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_10: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_201, [0, 0, 256, 256], -1.0);  view_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_17: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_10, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_56: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
        permute_179: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_56, [0, 1, 4, 3, 2]);  unsqueeze_56 = None
        permute_181: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_179, [0, 1, 4, 3, 2]);  permute_179 = None
        clone_34: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_205: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_34, [192, 768, 64]);  clone_34 = None
        bmm_5: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
        view_206: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_5, [48, 4, 256, 1, 64]);  bmm_5 = None
        permute_182: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_206, [0, 1, 2, 4, 3]);  view_206 = None
        view_207: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_182, [48, 4, 256, 64]);  permute_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_208: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_207, [4, 12, 1024, 64]);  view_207 = None
        permute_183: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_184: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_183, [1, 0, 2, 3]);  permute_183 = None
        clone_35: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_209: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_35, [1024, 4, 768]);  clone_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_185: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_209, [1, 0, 2]);  view_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_36: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_210: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_36, [4096, 768]);  clone_36 = None
        permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_11: "f32[4096, 768]" = torch.ops.aten.mm.default(view_210, permute_186);  view_210 = permute_186 = None
        view_211: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_11, [4, 1024, 768]);  mm_11 = None
        add_37: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_211, arg42_1);  view_211 = arg42_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_38: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_37, add_29);  add_37 = add_29 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_8: "f32[4, 1024, 1]" = var_mean_4[0]
        getitem_9: "f32[4, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_22: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_9);  add_38 = getitem_9 = None
        add_39: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_17: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_4);  sub_22 = rsqrt_4 = None
        mul_18: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg43_1);  mul_17 = arg43_1 = None
        add_40: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_18, arg44_1);  mul_18 = arg44_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_212: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_40, [4096, 768])
        permute_187: "f32[768, 3072]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        
        # No stacktrace found for following nodes
        mm_default_19: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_212, permute_187);  view_212 = permute_187 = None
        add_tensor_19: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_19, arg46_1);  mm_default_19 = arg46_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_213: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_19, [4, 1024, 3072]);  add_tensor_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_19: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
        mul_20: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476);  view_213 = None
        erf_2: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_41: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_41);  mul_19 = add_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_214: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_21, [4096, 3072]);  mul_21 = None
        permute_188: "f32[3072, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        
        # No stacktrace found for following nodes
        mm_default_18: "f32[4096, 768]" = torch.ops.aten.mm.default(view_214, permute_188);  view_214 = permute_188 = None
        add_tensor_18: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg48_1);  mm_default_18 = arg48_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_215: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [4, 1024, 768]);  add_tensor_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_42: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_215, add_40);  view_215 = add_40 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_10: "f32[4, 1024, 1]" = var_mean_5[0]
        getitem_11: "f32[4, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_72: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_73: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
        full_default_39: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_27: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_703: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1);  slice_703 = None
        slice_699: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_701: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_699, 3, 256, 9223372036854775807);  slice_699 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_23: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_11);  add_42 = getitem_11 = None
        add_43: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_22: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_5);  sub_23 = rsqrt_5 = None
        mul_23: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg49_1);  mul_22 = arg49_1 = None
        add_44: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_23, arg50_1);  mul_23 = arg50_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_189: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_44, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_39: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format)
        view_216: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_39, [4096, 768]);  clone_39 = None
        permute_190: "f32[768, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_12: "f32[4096, 768]" = torch.ops.aten.mm.default(view_216, permute_190);  view_216 = permute_190 = None
        view_217: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_12, [1024, 4, 768]);  mm_12 = None
        add_45: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_217, arg52_1);  view_217 = arg52_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_30: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_45, 8.0);  add_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_228: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_30, [1024, 4, 12, 64]);  div_30 = None
        permute_200: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_228, [1, 0, 2, 3]);  view_228 = None
        permute_201: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_200, [0, 2, 1, 3]);  permute_200 = None
        view_229: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_201, [48, 1024, 64]);  permute_201 = None
        view_230: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_229, [48, 2, 512, 64]);  view_229 = None
        as_strided_20: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_230, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_230 = None
        unsqueeze_59: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_20, 4);  as_strided_20 = None
        permute_202: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_59, [0, 1, 2, 4, 3]);  unsqueeze_59 = None
        permute_203: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_202, [0, 1, 2, 4, 3]);  permute_202 = None
        clone_42: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_231: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_42, [144, 512, 64]);  clone_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_40: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format)
        view_218: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_40, [4096, 768]);  clone_40 = None
        permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_13: "f32[4096, 768]" = torch.ops.aten.mm.default(view_218, permute_191);  view_218 = permute_191 = None
        view_219: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_13, [1024, 4, 768]);  mm_13 = None
        add_46: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_219, arg54_1);  view_219 = arg54_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_223: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_46, [1024, 4, 12, 64]);  add_46 = None
        permute_194: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_223, [1, 0, 2, 3]);  view_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_196: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_194, [0, 2, 1, 3]);  permute_194 = None
        view_225: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_196, [48, 1024, 64]);  permute_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_227: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_225, [48, 2, 512, 64]);  view_225 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_19: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_227, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_58: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
        permute_198: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_58, [0, 1, 4, 2, 3]);  unsqueeze_58 = None
        permute_204: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_198, [0, 1, 4, 3, 2]);  permute_198 = None
        clone_43: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_232: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_43, [144, 64, 512]);  clone_43 = None
        bmm_6: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
        view_233: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_6, [48, 3, 512, 1, 512]);  bmm_6 = None
        permute_205: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_233, [0, 1, 2, 4, 3]);  view_233 = None
        view_234: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_205, [48, 3, 512, 512]);  permute_205 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_12: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_234, [0, 0, 0, 1], 0.0);  view_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_235: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_12, [48, 3, 512, 513]);  constant_pad_nd_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_696: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_235, 2, 0, 256)
        slice_697: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_696, 3, 0, 257);  slice_696 = None
        copy_36: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_701, slice_697);  slice_701 = slice_697 = None
        
        # No stacktrace found for following nodes
        slice_tensor_30: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_scatter_default_66: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_30, copy_36, 3, 256, 9223372036854775807);  slice_tensor_30 = copy_36 = None
        slice_scatter_default_67: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_27, slice_scatter_default_66, 1, 0, -1);  full_27 = slice_scatter_default_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_63: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_67, 1, -1);  select_63 = None
        select_62: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_67, 1, -1)
        slice_717: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_62, 2, 256, 9223372036854775807);  select_62 = None
        select_60: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_235, 1, -1)
        slice_710: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_60, 1, 256, 9223372036854775807);  select_60 = None
        slice_711: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_710, 2, 0, 257);  slice_710 = None
        copy_37: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_717, slice_711);  slice_717 = slice_711 = None
        
        # No stacktrace found for following nodes
        select_int_12: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_67, 1, -1)
        slice_scatter_default_68: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_12, copy_37, 2, 256, 9223372036854775807);  select_int_12 = copy_37 = None
        select_scatter_default_12: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_67, slice_scatter_default_68, 1, -1);  slice_scatter_default_67 = slice_scatter_default_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_736: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_12, 1, 1, 9223372036854775807);  slice_736 = None
        slice_732: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_12, 1, 1, 9223372036854775807)
        slice_734: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_732, 3, 0, 256);  slice_732 = None
        slice_725: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_235, 2, -257, -1)
        slice_726: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_725, 3, 257, 9223372036854775807);  slice_725 = None
        copy_38: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_734, slice_726);  slice_734 = slice_726 = None
        
        # No stacktrace found for following nodes
        slice_tensor_31: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_12, 1, 1, 9223372036854775807)
        slice_scatter_default_69: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_31, copy_38, 3, 0, 256);  slice_tensor_31 = copy_38 = None
        slice_scatter_default_70: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_12, slice_scatter_default_69, 1, 1, 9223372036854775807);  select_scatter_default_12 = slice_scatter_default_69 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_68: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_70, 1, 0)
        slice_752: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_68, 1, 1, 256);  select_68 = slice_752 = None
        select_67: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_70, 1, 0)
        slice_749: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_67, 1, 1, 256);  select_67 = None
        slice_750: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_749, 2, 1, 256);  slice_749 = None
        select_65: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_235, 1, 0);  view_235 = None
        slice_743: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_65, 1, 0, 255);  select_65 = None
        slice_744: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_743, 2, -255, 9223372036854775807);  slice_743 = None
        copy_39: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_750, slice_744);  slice_750 = slice_744 = None
        
        # No stacktrace found for following nodes
        select_int_13: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_70, 1, 0)
        slice_tensor_32: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_13, 1, 1, 256)
        slice_scatter_default_71: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_32, copy_39, 2, 1, 256);  slice_tensor_32 = copy_39 = None
        slice_scatter_default_72: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_13, slice_scatter_default_71, 1, 1, 256);  select_int_13 = slice_scatter_default_71 = None
        select_scatter_default_13: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_70, slice_scatter_default_72, 1, 0);  slice_scatter_default_70 = slice_scatter_default_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_240: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_13, [4, 12, 1024, 513])
        permute_210: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        slice_775: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_210, 1, 0, 256);  slice_775 = None
        view_239: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_13, [4, 12, 1024, 513])
        permute_209: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
        slice_771: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_209, 1, 0, 256);  permute_209 = None
        slice_773: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_771, 3, 0, 257);  slice_771 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_12: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_60: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_12, -2);  iota_12 = None
        iota_13: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_61: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
        sub_25: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_60, unsqueeze_61);  unsqueeze_60 = unsqueeze_61 = None
        le_6: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_25, 0);  sub_25 = None
        full_default_30: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f32[256, 257]" = torch.ops.aten.where.self(le_6, full_default_30, full_default_31);  le_6 = full_default_30 = full_default_31 = None
        rev_12: "f32[256, 257]" = torch.ops.prims.rev.default(where_24, [0]);  where_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_62: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_12, 0);  rev_12 = None
        unsqueeze_63: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, 2);  unsqueeze_62 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_12: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_63, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_15: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_12, torch.bool);  expand_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_32: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_238: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_13, [4, 12, 1024, 513]);  select_scatter_default_13 = None
        permute_208: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        slice_763: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_208, 1, 0, 256);  permute_208 = None
        slice_765: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_763, 3, 0, 257);  slice_763 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_25: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_15, full_default_32, slice_765);  convert_element_type_15 = full_default_32 = slice_765 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_40: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_773, where_25);  slice_773 = where_25 = None
        
        # No stacktrace found for following nodes
        slice_tensor_33: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_210, 1, 0, 256)
        slice_scatter_default_73: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_33, copy_40, 3, 0, 257);  slice_tensor_33 = copy_40 = None
        slice_scatter_default_74: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_210, slice_scatter_default_73, 1, 0, 256);  permute_210 = slice_scatter_default_73 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_211: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_74, [0, 2, 1, 3]);  slice_scatter_default_74 = None
        view_241: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_211, [48, 4, 256, 513]);  permute_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_245: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_241, [4, 12, 1024, 513])
        permute_215: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        slice_798: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_215, 1, -256, 9223372036854775807);  slice_798 = None
        view_244: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_241, [4, 12, 1024, 513])
        permute_214: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        slice_794: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_214, 1, -256, 9223372036854775807);  permute_214 = None
        slice_796: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_794, 3, -257, 9223372036854775807);  slice_794 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_13: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_63, [1, 3]);  unsqueeze_63 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_13: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_13, [4, 256, 12, 257]);  rev_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_16: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_13, torch.bool);  expand_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_33: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_243: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_241, [4, 12, 1024, 513]);  view_241 = None
        permute_213: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        slice_786: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_213, 1, -256, 9223372036854775807);  permute_213 = None
        slice_788: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_786, 3, -257, 9223372036854775807);  slice_786 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_26: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_16, full_default_33, slice_788);  convert_element_type_16 = full_default_33 = slice_788 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_41: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_796, where_26);  slice_796 = where_26 = None
        
        # No stacktrace found for following nodes
        slice_tensor_34: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_215, 1, -256, 9223372036854775807)
        slice_scatter_default_75: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_34, copy_41, 3, -257, 9223372036854775807);  slice_tensor_34 = copy_41 = None
        slice_scatter_default_76: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_215, slice_scatter_default_75, 1, -256, 9223372036854775807);  permute_215 = slice_scatter_default_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_216: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_76, [0, 2, 1, 3]);  slice_scatter_default_76 = None
        view_246: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_216, [48, 4, 256, 513]);  permute_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_266: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_246, [4, 12, 1024, 513]);  view_246 = None
        permute_234: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_32: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_815: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1);  slice_815 = None
        slice_811: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1)
        slice_813: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_811, 3, 256, 9223372036854775807);  slice_811 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_31: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_218: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_31, [0, 2, 1, 3]);  full_31 = None
        view_248: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_218, [4, 1024, 1]);  permute_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_250: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_248, [4, 2, 512, 1]);  view_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_21: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_250, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_66: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_21, 4);  as_strided_21 = None
        permute_220: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_66, [0, 1, 2, 4, 3]);  unsqueeze_66 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_34: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_3: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_64: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_3, 2);  ne_3 = None
        unsqueeze_65: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 3);  unsqueeze_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_17: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_65, torch.float32)
        where_27: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_65, full_default_34, convert_element_type_17);  unsqueeze_65 = full_default_34 = convert_element_type_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_219: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_27, [0, 2, 1, 3]);  where_27 = None
        view_249: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_219, [4, 1024, 1]);  permute_219 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_251: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_249, [4, 2, 512, 1]);  view_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_22: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_251, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_67: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_22, 4);  as_strided_22 = None
        permute_221: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_67, [0, 1, 4, 2, 3]);  unsqueeze_67 = None
        mul_24: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_220, permute_221);  permute_220 = permute_221 = None
        view_252: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_24, [4, 3, 512, 512]);  mul_24 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_13: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_252, [0, 0, 0, 1], 0.0);  view_252 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_253: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_13, [4, 3, 512, 513]);  constant_pad_nd_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_808: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_253, 2, 0, 256)
        slice_809: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_808, 3, 0, 257);  slice_808 = None
        copy_42: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_813, slice_809);  slice_813 = slice_809 = None
        
        # No stacktrace found for following nodes
        slice_tensor_35: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_32, 1, 0, -1)
        slice_scatter_default_77: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_35, copy_42, 3, 256, 9223372036854775807);  slice_tensor_35 = copy_42 = None
        slice_scatter_default_78: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_32, slice_scatter_default_77, 1, 0, -1);  full_32 = slice_scatter_default_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_73: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_78, 1, -1);  select_73 = None
        select_72: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_78, 1, -1)
        slice_829: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_72, 2, 256, 9223372036854775807);  select_72 = None
        select_70: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_253, 1, -1)
        slice_822: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_70, 1, 256, 9223372036854775807);  select_70 = None
        slice_823: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_822, 2, 0, 257);  slice_822 = None
        copy_43: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_829, slice_823);  slice_829 = slice_823 = None
        
        # No stacktrace found for following nodes
        select_int_14: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_78, 1, -1)
        slice_scatter_default_79: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_14, copy_43, 2, 256, 9223372036854775807);  select_int_14 = copy_43 = None
        select_scatter_default_14: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_78, slice_scatter_default_79, 1, -1);  slice_scatter_default_78 = slice_scatter_default_79 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_848: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_14, 1, 1, 9223372036854775807);  slice_848 = None
        slice_844: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_14, 1, 1, 9223372036854775807)
        slice_846: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_844, 3, 0, 256);  slice_844 = None
        slice_837: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_253, 2, -257, -1)
        slice_838: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_837, 3, 257, 9223372036854775807);  slice_837 = None
        copy_44: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_846, slice_838);  slice_846 = slice_838 = None
        
        # No stacktrace found for following nodes
        slice_tensor_36: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_14, 1, 1, 9223372036854775807)
        slice_scatter_default_80: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_36, copy_44, 3, 0, 256);  slice_tensor_36 = copy_44 = None
        slice_scatter_default_81: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_14, slice_scatter_default_80, 1, 1, 9223372036854775807);  select_scatter_default_14 = slice_scatter_default_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_78: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_81, 1, 0)
        slice_864: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_78, 1, 1, 256);  select_78 = slice_864 = None
        select_77: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_81, 1, 0)
        slice_861: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_77, 1, 1, 256);  select_77 = None
        slice_862: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_861, 2, 1, 256);  slice_861 = None
        select_75: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_253, 1, 0);  view_253 = None
        slice_855: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_75, 1, 0, 255);  select_75 = None
        slice_856: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_855, 2, -255, 9223372036854775807);  slice_855 = None
        copy_45: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_862, slice_856);  slice_862 = slice_856 = None
        
        # No stacktrace found for following nodes
        select_int_15: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_81, 1, 0)
        slice_tensor_37: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_15, 1, 1, 256)
        slice_scatter_default_82: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_37, copy_45, 2, 1, 256);  slice_tensor_37 = copy_45 = None
        slice_scatter_default_83: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_15, slice_scatter_default_82, 1, 1, 256);  select_int_15 = slice_scatter_default_82 = None
        select_scatter_default_15: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_81, slice_scatter_default_83, 1, 0);  slice_scatter_default_81 = slice_scatter_default_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_258: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_15, [4, 1, 1024, 513])
        permute_226: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        slice_887: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_226, 1, 0, 256);  slice_887 = None
        view_257: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_15, [4, 1, 1024, 513])
        permute_225: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        slice_883: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_225, 1, 0, 256);  permute_225 = None
        slice_885: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_883, 3, 0, 257);  slice_883 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_14: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_68: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_14, -2);  iota_14 = None
        iota_15: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_69: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
        sub_27: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_68, unsqueeze_69);  unsqueeze_68 = unsqueeze_69 = None
        le_7: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_27, 0);  sub_27 = None
        full_default_35: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f32[256, 257]" = torch.ops.aten.where.self(le_7, full_default_35, full_default_36);  le_7 = full_default_35 = full_default_36 = None
        rev_14: "f32[256, 257]" = torch.ops.prims.rev.default(where_28, [0]);  where_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_70: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_14, 0);  rev_14 = None
        unsqueeze_71: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, 2);  unsqueeze_70 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_14: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_71, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_18: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_14, torch.bool);  expand_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_37: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_256: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_15, [4, 1, 1024, 513]);  select_scatter_default_15 = None
        permute_224: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        slice_875: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_224, 1, 0, 256);  permute_224 = None
        slice_877: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_875, 3, 0, 257);  slice_875 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_29: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_18, full_default_37, slice_877);  convert_element_type_18 = full_default_37 = slice_877 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_46: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_885, where_29);  slice_885 = where_29 = None
        
        # No stacktrace found for following nodes
        slice_tensor_38: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_226, 1, 0, 256)
        slice_scatter_default_84: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_38, copy_46, 3, 0, 257);  slice_tensor_38 = copy_46 = None
        slice_scatter_default_85: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_226, slice_scatter_default_84, 1, 0, 256);  permute_226 = slice_scatter_default_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_227: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_85, [0, 2, 1, 3]);  slice_scatter_default_85 = None
        view_259: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_227, [4, 4, 256, 513]);  permute_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_263: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_259, [4, 1, 1024, 513])
        permute_231: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
        slice_910: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_231, 1, -256, 9223372036854775807);  slice_910 = None
        view_262: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_259, [4, 1, 1024, 513])
        permute_230: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        slice_906: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_230, 1, -256, 9223372036854775807);  permute_230 = None
        slice_908: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_906, 3, -257, 9223372036854775807);  slice_906 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_15: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_71, [1, 3]);  unsqueeze_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_15: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_15, [4, 256, 1, 257]);  rev_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_19: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_15, torch.bool);  expand_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_38: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_261: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_259, [4, 1, 1024, 513]);  view_259 = None
        permute_229: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        slice_898: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_229, 1, -256, 9223372036854775807);  permute_229 = None
        slice_900: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_898, 3, -257, 9223372036854775807);  slice_898 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_30: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_19, full_default_38, slice_900);  convert_element_type_19 = full_default_38 = slice_900 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_47: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_908, where_30);  slice_908 = where_30 = None
        
        # No stacktrace found for following nodes
        slice_tensor_39: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_231, 1, -256, 9223372036854775807)
        slice_scatter_default_86: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_39, copy_47, 3, -257, 9223372036854775807);  slice_tensor_39 = copy_47 = None
        slice_scatter_default_87: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_231, slice_scatter_default_86, 1, -256, 9223372036854775807);  permute_231 = slice_scatter_default_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_232: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_87, [0, 2, 1, 3]);  slice_scatter_default_87 = None
        view_264: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_232, [4, 4, 256, 513]);  permute_232 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_267: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_264, [4, 1, 1024, 513]);  view_264 = None
        permute_235: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        add_50: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_234, permute_235);  permute_234 = permute_235 = None
        permute_236: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_50, [0, 2, 1, 3]);  add_50 = None
        view_269: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_236, [48, 4, 256, 513]);  permute_236 = None
        view_270: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_269, [4, 12, 1024, 513]);  view_269 = None
        permute_237: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_44: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        amax_3: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_44, [-1], True)
        sub_28: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_44, amax_3);  clone_44 = amax_3 = None
        exp_3: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_4: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_37: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_31: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_73, full_default_39, div_37);  unsqueeze_73 = full_default_39 = div_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_239: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_31, [0, 2, 1, 3]);  where_31 = None
        clone_46: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_272: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_46, [48, 4, 256, 513]);  clone_46 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_15: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_272, [0, 257], 0.0);  view_272 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_274: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_15, [48, 4, -1]);  constant_pad_nd_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_920: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_274, 2, 0, -256);  view_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_275: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_920, [48, 4, 256, 769]);  slice_920 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_924: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_275, 3, 0, -1);  view_275 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_74: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_924, 4);  slice_924 = None
        permute_241: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_74, [0, 1, 2, 4, 3]);  unsqueeze_74 = None
        permute_243: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_241, [0, 1, 2, 4, 3]);  permute_241 = None
        view_276: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_243, [192, 256, 768]);  permute_243 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_41: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_220: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_41, [4096, 768]);  clone_41 = None
        permute_192: "f32[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_14: "f32[4096, 768]" = torch.ops.aten.mm.default(view_220, permute_192);  view_220 = permute_192 = None
        view_221: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_14, [1024, 4, 768]);  mm_14 = None
        add_47: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_221, arg56_1);  view_221 = arg56_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_271: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_47, [1024, 4, 12, 64]);  add_47 = None
        permute_238: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_271, [1, 0, 2, 3]);  view_271 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_240: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_238, [0, 2, 1, 3]);  permute_238 = None
        view_273: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_240, [48, 1024, 64]);  permute_240 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_14: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_273, [0, 0, 256, 256], -1.0);  view_273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_23: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_14, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_75: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
        permute_242: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
        permute_244: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_242, [0, 1, 4, 3, 2]);  permute_242 = None
        clone_47: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_277: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_47, [192, 768, 64]);  clone_47 = None
        bmm_7: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_276, view_277);  view_276 = view_277 = None
        view_278: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_7, [48, 4, 256, 1, 64]);  bmm_7 = None
        permute_245: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_278, [0, 1, 2, 4, 3]);  view_278 = None
        view_279: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_245, [48, 4, 256, 64]);  permute_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_280: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_279, [4, 12, 1024, 64]);  view_279 = None
        permute_246: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_247: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_246, [1, 0, 2, 3]);  permute_246 = None
        clone_48: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_281: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_48, [1024, 4, 768]);  clone_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_248: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_281, [1, 0, 2]);  view_281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_49: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_282: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_49, [4096, 768]);  clone_49 = None
        permute_249: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_15: "f32[4096, 768]" = torch.ops.aten.mm.default(view_282, permute_249);  view_282 = permute_249 = None
        view_283: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_15, [4, 1024, 768]);  mm_15 = None
        add_52: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_283, arg58_1);  view_283 = arg58_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_52, add_44);  add_52 = add_44 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_12: "f32[4, 1024, 1]" = var_mean_6[0]
        getitem_13: "f32[4, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_30: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_13);  add_53 = getitem_13 = None
        add_54: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_25: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_6);  sub_30 = rsqrt_6 = None
        mul_26: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg59_1);  mul_25 = arg59_1 = None
        add_55: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_26, arg60_1);  mul_26 = arg60_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_284: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_55, [4096, 768])
        permute_250: "f32[768, 3072]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        
        # No stacktrace found for following nodes
        mm_default_17: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_284, permute_250);  view_284 = permute_250 = None
        add_tensor_17: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_17, arg62_1);  mm_default_17 = arg62_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_285: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_17, [4, 1024, 3072]);  add_tensor_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_27: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
        mul_28: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_285, 0.7071067811865476);  view_285 = None
        erf_3: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_56: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_29: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_56);  mul_27 = add_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_286: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_29, [4096, 3072]);  mul_29 = None
        permute_251: "f32[3072, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        
        # No stacktrace found for following nodes
        mm_default_16: "f32[4096, 768]" = torch.ops.aten.mm.default(view_286, permute_251);  view_286 = permute_251 = None
        add_tensor_16: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_16, arg64_1);  mm_default_16 = arg64_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_287: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_16, [4, 1024, 768]);  add_tensor_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_287, add_55);  view_287 = add_55 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_14: "f32[4, 1024, 1]" = var_mean_7[0]
        getitem_15: "f32[4, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_91: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_92: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 3);  unsqueeze_91 = None
        full_default_49: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_36: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_934: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1);  slice_934 = None
        slice_930: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1)
        slice_932: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_930, 3, 256, 9223372036854775807);  slice_930 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_31: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_15);  add_57 = getitem_15 = None
        add_58: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_30: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_7);  sub_31 = rsqrt_7 = None
        mul_31: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg65_1);  mul_30 = arg65_1 = None
        add_59: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_31, arg66_1);  mul_31 = arg66_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_252: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_59, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_52: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format)
        view_288: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_52, [4096, 768]);  clone_52 = None
        permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_16: "f32[4096, 768]" = torch.ops.aten.mm.default(view_288, permute_253);  view_288 = permute_253 = None
        view_289: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_16, [1024, 4, 768]);  mm_16 = None
        add_60: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_289, arg68_1);  view_289 = arg68_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_40: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_60, 8.0);  add_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_300: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_40, [1024, 4, 12, 64]);  div_40 = None
        permute_263: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_300, [1, 0, 2, 3]);  view_300 = None
        permute_264: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_263, [0, 2, 1, 3]);  permute_263 = None
        view_301: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_264, [48, 1024, 64]);  permute_264 = None
        view_302: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_301, [48, 2, 512, 64]);  view_301 = None
        as_strided_26: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_302, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_302 = None
        unsqueeze_78: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
        permute_265: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_78, [0, 1, 2, 4, 3]);  unsqueeze_78 = None
        permute_266: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_265, [0, 1, 2, 4, 3]);  permute_265 = None
        clone_55: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
        view_303: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_55, [144, 512, 64]);  clone_55 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_53: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format)
        view_290: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_53, [4096, 768]);  clone_53 = None
        permute_254: "f32[768, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_17: "f32[4096, 768]" = torch.ops.aten.mm.default(view_290, permute_254);  view_290 = permute_254 = None
        view_291: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_17, [1024, 4, 768]);  mm_17 = None
        add_61: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_291, arg70_1);  view_291 = arg70_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_295: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_61, [1024, 4, 12, 64]);  add_61 = None
        permute_257: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_295, [1, 0, 2, 3]);  view_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_259: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_257, [0, 2, 1, 3]);  permute_257 = None
        view_297: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_259, [48, 1024, 64]);  permute_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_299: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_297, [48, 2, 512, 64]);  view_297 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_25: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_299, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_299 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_77: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_25, 4);  as_strided_25 = None
        permute_261: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_77, [0, 1, 4, 2, 3]);  unsqueeze_77 = None
        permute_267: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_261, [0, 1, 4, 3, 2]);  permute_261 = None
        clone_56: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_304: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_56, [144, 64, 512]);  clone_56 = None
        bmm_8: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_303, view_304);  view_303 = view_304 = None
        view_305: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_8, [48, 3, 512, 1, 512]);  bmm_8 = None
        permute_268: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_305, [0, 1, 2, 4, 3]);  view_305 = None
        view_306: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_268, [48, 3, 512, 512]);  permute_268 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_16: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_306, [0, 0, 0, 1], 0.0);  view_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_307: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_16, [48, 3, 512, 513]);  constant_pad_nd_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_927: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_307, 2, 0, 256)
        slice_928: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_927, 3, 0, 257);  slice_927 = None
        copy_48: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_932, slice_928);  slice_932 = slice_928 = None
        
        # No stacktrace found for following nodes
        slice_tensor_40: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_36, 1, 0, -1)
        slice_scatter_default_88: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_40, copy_48, 3, 256, 9223372036854775807);  slice_tensor_40 = copy_48 = None
        slice_scatter_default_89: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_36, slice_scatter_default_88, 1, 0, -1);  full_36 = slice_scatter_default_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_83: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_89, 1, -1);  select_83 = None
        select_82: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_89, 1, -1)
        slice_948: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_82, 2, 256, 9223372036854775807);  select_82 = None
        select_80: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_307, 1, -1)
        slice_941: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_80, 1, 256, 9223372036854775807);  select_80 = None
        slice_942: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_941, 2, 0, 257);  slice_941 = None
        copy_49: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_948, slice_942);  slice_948 = slice_942 = None
        
        # No stacktrace found for following nodes
        select_int_16: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_89, 1, -1)
        slice_scatter_default_90: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_16, copy_49, 2, 256, 9223372036854775807);  select_int_16 = copy_49 = None
        select_scatter_default_16: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_89, slice_scatter_default_90, 1, -1);  slice_scatter_default_89 = slice_scatter_default_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_967: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_16, 1, 1, 9223372036854775807);  slice_967 = None
        slice_963: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_16, 1, 1, 9223372036854775807)
        slice_965: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_963, 3, 0, 256);  slice_963 = None
        slice_956: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_307, 2, -257, -1)
        slice_957: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_956, 3, 257, 9223372036854775807);  slice_956 = None
        copy_50: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_965, slice_957);  slice_965 = slice_957 = None
        
        # No stacktrace found for following nodes
        slice_tensor_41: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_16, 1, 1, 9223372036854775807)
        slice_scatter_default_91: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_41, copy_50, 3, 0, 256);  slice_tensor_41 = copy_50 = None
        slice_scatter_default_92: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_16, slice_scatter_default_91, 1, 1, 9223372036854775807);  select_scatter_default_16 = slice_scatter_default_91 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_88: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_92, 1, 0)
        slice_983: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_88, 1, 1, 256);  select_88 = slice_983 = None
        select_87: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_92, 1, 0)
        slice_980: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_87, 1, 1, 256);  select_87 = None
        slice_981: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_980, 2, 1, 256);  slice_980 = None
        select_85: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_307, 1, 0);  view_307 = None
        slice_974: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_85, 1, 0, 255);  select_85 = None
        slice_975: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_974, 2, -255, 9223372036854775807);  slice_974 = None
        copy_51: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_981, slice_975);  slice_981 = slice_975 = None
        
        # No stacktrace found for following nodes
        select_int_17: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_92, 1, 0)
        slice_tensor_42: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_17, 1, 1, 256)
        slice_scatter_default_93: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_42, copy_51, 2, 1, 256);  slice_tensor_42 = copy_51 = None
        slice_scatter_default_94: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_17, slice_scatter_default_93, 1, 1, 256);  select_int_17 = slice_scatter_default_93 = None
        select_scatter_default_17: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_92, slice_scatter_default_94, 1, 0);  slice_scatter_default_92 = slice_scatter_default_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_312: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_17, [4, 12, 1024, 513])
        permute_273: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
        slice_1006: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_273, 1, 0, 256);  slice_1006 = None
        view_311: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_17, [4, 12, 1024, 513])
        permute_272: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
        slice_1002: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_272, 1, 0, 256);  permute_272 = None
        slice_1004: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1002, 3, 0, 257);  slice_1002 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_16: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_79: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_16, -2);  iota_16 = None
        iota_17: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_80: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
        sub_33: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_79, unsqueeze_80);  unsqueeze_79 = unsqueeze_80 = None
        le_8: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_33, 0);  sub_33 = None
        full_default_40: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: "f32[256, 257]" = torch.ops.aten.where.self(le_8, full_default_40, full_default_41);  le_8 = full_default_40 = full_default_41 = None
        rev_16: "f32[256, 257]" = torch.ops.prims.rev.default(where_32, [0]);  where_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_81: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_16, 0);  rev_16 = None
        unsqueeze_82: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 2);  unsqueeze_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_16: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_82, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_20: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_16, torch.bool);  expand_16 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_42: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_310: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_17, [4, 12, 1024, 513]);  select_scatter_default_17 = None
        permute_271: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        slice_994: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_271, 1, 0, 256);  permute_271 = None
        slice_996: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_994, 3, 0, 257);  slice_994 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_33: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_20, full_default_42, slice_996);  convert_element_type_20 = full_default_42 = slice_996 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_52: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1004, where_33);  slice_1004 = where_33 = None
        
        # No stacktrace found for following nodes
        slice_tensor_43: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_273, 1, 0, 256)
        slice_scatter_default_95: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_43, copy_52, 3, 0, 257);  slice_tensor_43 = copy_52 = None
        slice_scatter_default_96: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_273, slice_scatter_default_95, 1, 0, 256);  permute_273 = slice_scatter_default_95 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_274: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_96, [0, 2, 1, 3]);  slice_scatter_default_96 = None
        view_313: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_274, [48, 4, 256, 513]);  permute_274 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_317: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_313, [4, 12, 1024, 513])
        permute_278: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
        slice_1029: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_278, 1, -256, 9223372036854775807);  slice_1029 = None
        view_316: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_313, [4, 12, 1024, 513])
        permute_277: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        slice_1025: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_277, 1, -256, 9223372036854775807);  permute_277 = None
        slice_1027: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1025, 3, -257, 9223372036854775807);  slice_1025 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_17: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_82, [1, 3]);  unsqueeze_82 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_17: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_17, [4, 256, 12, 257]);  rev_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_21: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_17, torch.bool);  expand_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_43: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_315: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_313, [4, 12, 1024, 513]);  view_313 = None
        permute_276: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        slice_1017: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_276, 1, -256, 9223372036854775807);  permute_276 = None
        slice_1019: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1017, 3, -257, 9223372036854775807);  slice_1017 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_34: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_21, full_default_43, slice_1019);  convert_element_type_21 = full_default_43 = slice_1019 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_53: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1027, where_34);  slice_1027 = where_34 = None
        
        # No stacktrace found for following nodes
        slice_tensor_44: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_278, 1, -256, 9223372036854775807)
        slice_scatter_default_97: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_44, copy_53, 3, -257, 9223372036854775807);  slice_tensor_44 = copy_53 = None
        slice_scatter_default_98: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_278, slice_scatter_default_97, 1, -256, 9223372036854775807);  permute_278 = slice_scatter_default_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_279: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_98, [0, 2, 1, 3]);  slice_scatter_default_98 = None
        view_318: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_279, [48, 4, 256, 513]);  permute_279 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_338: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_318, [4, 12, 1024, 513]);  view_318 = None
        permute_297: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_41: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1046: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1);  slice_1046 = None
        slice_1042: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1)
        slice_1044: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1042, 3, 256, 9223372036854775807);  slice_1042 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_40: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_281: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_40, [0, 2, 1, 3]);  full_40 = None
        view_320: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_281, [4, 1024, 1]);  permute_281 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_322: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_320, [4, 2, 512, 1]);  view_320 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_27: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_322, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_85: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_27, 4);  as_strided_27 = None
        permute_283: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_85, [0, 1, 2, 4, 3]);  unsqueeze_85 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_44: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_4: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_83: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_4, 2);  ne_4 = None
        unsqueeze_84: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 3);  unsqueeze_83 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_22: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_84, torch.float32)
        where_35: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_84, full_default_44, convert_element_type_22);  unsqueeze_84 = full_default_44 = convert_element_type_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_282: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_35, [0, 2, 1, 3]);  where_35 = None
        view_321: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_282, [4, 1024, 1]);  permute_282 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_323: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_321, [4, 2, 512, 1]);  view_321 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_28: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_323, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_323 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_86: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_28, 4);  as_strided_28 = None
        permute_284: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_86, [0, 1, 4, 2, 3]);  unsqueeze_86 = None
        mul_32: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_283, permute_284);  permute_283 = permute_284 = None
        view_324: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_32, [4, 3, 512, 512]);  mul_32 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_17: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_324, [0, 0, 0, 1], 0.0);  view_324 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_325: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_17, [4, 3, 512, 513]);  constant_pad_nd_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1039: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_325, 2, 0, 256)
        slice_1040: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1039, 3, 0, 257);  slice_1039 = None
        copy_54: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1044, slice_1040);  slice_1044 = slice_1040 = None
        
        # No stacktrace found for following nodes
        slice_tensor_45: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_41, 1, 0, -1)
        slice_scatter_default_99: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_45, copy_54, 3, 256, 9223372036854775807);  slice_tensor_45 = copy_54 = None
        slice_scatter_default_100: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_41, slice_scatter_default_99, 1, 0, -1);  full_41 = slice_scatter_default_99 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_93: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_100, 1, -1);  select_93 = None
        select_92: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_100, 1, -1)
        slice_1060: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_92, 2, 256, 9223372036854775807);  select_92 = None
        select_90: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_325, 1, -1)
        slice_1053: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_90, 1, 256, 9223372036854775807);  select_90 = None
        slice_1054: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1053, 2, 0, 257);  slice_1053 = None
        copy_55: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_1060, slice_1054);  slice_1060 = slice_1054 = None
        
        # No stacktrace found for following nodes
        select_int_18: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_100, 1, -1)
        slice_scatter_default_101: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_18, copy_55, 2, 256, 9223372036854775807);  select_int_18 = copy_55 = None
        select_scatter_default_18: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_100, slice_scatter_default_101, 1, -1);  slice_scatter_default_100 = slice_scatter_default_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1079: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_18, 1, 1, 9223372036854775807);  slice_1079 = None
        slice_1075: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_18, 1, 1, 9223372036854775807)
        slice_1077: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1075, 3, 0, 256);  slice_1075 = None
        slice_1068: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_325, 2, -257, -1)
        slice_1069: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1068, 3, 257, 9223372036854775807);  slice_1068 = None
        copy_56: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1077, slice_1069);  slice_1077 = slice_1069 = None
        
        # No stacktrace found for following nodes
        slice_tensor_46: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_18, 1, 1, 9223372036854775807)
        slice_scatter_default_102: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_46, copy_56, 3, 0, 256);  slice_tensor_46 = copy_56 = None
        slice_scatter_default_103: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_18, slice_scatter_default_102, 1, 1, 9223372036854775807);  select_scatter_default_18 = slice_scatter_default_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_98: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_103, 1, 0)
        slice_1095: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_98, 1, 1, 256);  select_98 = slice_1095 = None
        select_97: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_103, 1, 0)
        slice_1092: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_97, 1, 1, 256);  select_97 = None
        slice_1093: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1092, 2, 1, 256);  slice_1092 = None
        select_95: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_325, 1, 0);  view_325 = None
        slice_1086: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_95, 1, 0, 255);  select_95 = None
        slice_1087: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1086, 2, -255, 9223372036854775807);  slice_1086 = None
        copy_57: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_1093, slice_1087);  slice_1093 = slice_1087 = None
        
        # No stacktrace found for following nodes
        select_int_19: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_103, 1, 0)
        slice_tensor_47: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_19, 1, 1, 256)
        slice_scatter_default_104: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_47, copy_57, 2, 1, 256);  slice_tensor_47 = copy_57 = None
        slice_scatter_default_105: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_19, slice_scatter_default_104, 1, 1, 256);  select_int_19 = slice_scatter_default_104 = None
        select_scatter_default_19: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_103, slice_scatter_default_105, 1, 0);  slice_scatter_default_103 = slice_scatter_default_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_330: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_19, [4, 1, 1024, 513])
        permute_289: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        slice_1118: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_289, 1, 0, 256);  slice_1118 = None
        view_329: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_19, [4, 1, 1024, 513])
        permute_288: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
        slice_1114: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_288, 1, 0, 256);  permute_288 = None
        slice_1116: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1114, 3, 0, 257);  slice_1114 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_18: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_87: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_18, -2);  iota_18 = None
        iota_19: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_88: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
        sub_35: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_87, unsqueeze_88);  unsqueeze_87 = unsqueeze_88 = None
        le_9: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_35, 0);  sub_35 = None
        full_default_45: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36: "f32[256, 257]" = torch.ops.aten.where.self(le_9, full_default_45, full_default_46);  le_9 = full_default_45 = full_default_46 = None
        rev_18: "f32[256, 257]" = torch.ops.prims.rev.default(where_36, [0]);  where_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_89: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_18, 0);  rev_18 = None
        unsqueeze_90: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_18: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_90, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_23: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_18, torch.bool);  expand_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_47: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_328: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_19, [4, 1, 1024, 513]);  select_scatter_default_19 = None
        permute_287: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        slice_1106: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_287, 1, 0, 256);  permute_287 = None
        slice_1108: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1106, 3, 0, 257);  slice_1106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_37: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_23, full_default_47, slice_1108);  convert_element_type_23 = full_default_47 = slice_1108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_58: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1116, where_37);  slice_1116 = where_37 = None
        
        # No stacktrace found for following nodes
        slice_tensor_48: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_289, 1, 0, 256)
        slice_scatter_default_106: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_48, copy_58, 3, 0, 257);  slice_tensor_48 = copy_58 = None
        slice_scatter_default_107: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_289, slice_scatter_default_106, 1, 0, 256);  permute_289 = slice_scatter_default_106 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_290: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_107, [0, 2, 1, 3]);  slice_scatter_default_107 = None
        view_331: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_290, [4, 4, 256, 513]);  permute_290 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_335: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_331, [4, 1, 1024, 513])
        permute_294: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
        slice_1141: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_294, 1, -256, 9223372036854775807);  slice_1141 = None
        view_334: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_331, [4, 1, 1024, 513])
        permute_293: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
        slice_1137: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_293, 1, -256, 9223372036854775807);  permute_293 = None
        slice_1139: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1137, 3, -257, 9223372036854775807);  slice_1137 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_19: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_90, [1, 3]);  unsqueeze_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_19: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_19, [4, 256, 1, 257]);  rev_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_24: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_19, torch.bool);  expand_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_48: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_333: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_331, [4, 1, 1024, 513]);  view_331 = None
        permute_292: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
        slice_1129: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_292, 1, -256, 9223372036854775807);  permute_292 = None
        slice_1131: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1129, 3, -257, 9223372036854775807);  slice_1129 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_38: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_24, full_default_48, slice_1131);  convert_element_type_24 = full_default_48 = slice_1131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_59: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1139, where_38);  slice_1139 = where_38 = None
        
        # No stacktrace found for following nodes
        slice_tensor_49: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_294, 1, -256, 9223372036854775807)
        slice_scatter_default_108: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_49, copy_59, 3, -257, 9223372036854775807);  slice_tensor_49 = copy_59 = None
        slice_scatter_default_109: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_294, slice_scatter_default_108, 1, -256, 9223372036854775807);  permute_294 = slice_scatter_default_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_295: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_109, [0, 2, 1, 3]);  slice_scatter_default_109 = None
        view_336: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_295, [4, 4, 256, 513]);  permute_295 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_339: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_336, [4, 1, 1024, 513]);  view_336 = None
        permute_298: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        add_65: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_297, permute_298);  permute_297 = permute_298 = None
        permute_299: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_65, [0, 2, 1, 3]);  add_65 = None
        view_341: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_299, [48, 4, 256, 513]);  permute_299 = None
        view_342: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_341, [4, 12, 1024, 513]);  view_341 = None
        permute_300: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_57: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
        amax_4: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_57, [-1], True)
        sub_36: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_57, amax_4);  clone_57 = amax_4 = None
        exp_4: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_5: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_47: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_39: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_92, full_default_49, div_47);  unsqueeze_92 = full_default_49 = div_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_302: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_39, [0, 2, 1, 3]);  where_39 = None
        clone_59: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_344: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_59, [48, 4, 256, 513]);  clone_59 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_19: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_344, [0, 257], 0.0);  view_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_346: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_19, [48, 4, -1]);  constant_pad_nd_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1151: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_346, 2, 0, -256);  view_346 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_347: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1151, [48, 4, 256, 769]);  slice_1151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1155: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_347, 3, 0, -1);  view_347 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_93: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1155, 4);  slice_1155 = None
        permute_304: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_93, [0, 1, 2, 4, 3]);  unsqueeze_93 = None
        permute_306: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_304, [0, 1, 2, 4, 3]);  permute_304 = None
        view_348: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_306, [192, 256, 768]);  permute_306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_54: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_292: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_54, [4096, 768]);  clone_54 = None
        permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_18: "f32[4096, 768]" = torch.ops.aten.mm.default(view_292, permute_255);  view_292 = permute_255 = None
        view_293: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_18, [1024, 4, 768]);  mm_18 = None
        add_62: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_293, arg72_1);  view_293 = arg72_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_343: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_62, [1024, 4, 12, 64]);  add_62 = None
        permute_301: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_343, [1, 0, 2, 3]);  view_343 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_303: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_301, [0, 2, 1, 3]);  permute_301 = None
        view_345: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_303, [48, 1024, 64]);  permute_303 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_18: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_345, [0, 0, 256, 256], -1.0);  view_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_29: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_18, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_18 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_94: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_29, 4);  as_strided_29 = None
        permute_305: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_94, [0, 1, 4, 3, 2]);  unsqueeze_94 = None
        permute_307: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_305, [0, 1, 4, 3, 2]);  permute_305 = None
        clone_60: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
        view_349: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_60, [192, 768, 64]);  clone_60 = None
        bmm_9: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_348, view_349);  view_348 = view_349 = None
        view_350: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_9, [48, 4, 256, 1, 64]);  bmm_9 = None
        permute_308: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_350, [0, 1, 2, 4, 3]);  view_350 = None
        view_351: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_308, [48, 4, 256, 64]);  permute_308 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_352: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_351, [4, 12, 1024, 64]);  view_351 = None
        permute_309: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_310: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_309, [1, 0, 2, 3]);  permute_309 = None
        clone_61: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        view_353: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_61, [1024, 4, 768]);  clone_61 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_311: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_353, [1, 0, 2]);  view_353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_62: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_311, memory_format = torch.contiguous_format);  permute_311 = None
        view_354: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_62, [4096, 768]);  clone_62 = None
        permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_19: "f32[4096, 768]" = torch.ops.aten.mm.default(view_354, permute_312);  view_354 = permute_312 = None
        view_355: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_19, [4, 1024, 768]);  mm_19 = None
        add_67: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_355, arg74_1);  view_355 = arg74_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_68: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_67, add_59);  add_67 = add_59 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_16: "f32[4, 1024, 1]" = var_mean_8[0]
        getitem_17: "f32[4, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_38: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_17);  add_68 = getitem_17 = None
        add_69: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_33: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_8);  sub_38 = rsqrt_8 = None
        mul_34: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg75_1);  mul_33 = arg75_1 = None
        add_70: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_34, arg76_1);  mul_34 = arg76_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_356: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_70, [4096, 768])
        permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        
        # No stacktrace found for following nodes
        mm_default_15: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_356, permute_313);  view_356 = permute_313 = None
        add_tensor_15: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_15, arg78_1);  mm_default_15 = arg78_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_357: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_15, [4, 1024, 3072]);  add_tensor_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_35: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.5)
        mul_36: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_357, 0.7071067811865476);  view_357 = None
        erf_4: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_71: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_37: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_71);  mul_35 = add_71 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_358: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_37, [4096, 3072]);  mul_37 = None
        permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        
        # No stacktrace found for following nodes
        mm_default_14: "f32[4096, 768]" = torch.ops.aten.mm.default(view_358, permute_314);  view_358 = permute_314 = None
        add_tensor_14: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_14, arg80_1);  mm_default_14 = arg80_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_359: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [4, 1024, 768]);  add_tensor_14 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_72: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_359, add_70);  view_359 = add_70 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_18: "f32[4, 1024, 1]" = var_mean_9[0]
        getitem_19: "f32[4, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_110: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_111: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 3);  unsqueeze_110 = None
        full_default_59: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_45: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1165: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1);  slice_1165 = None
        slice_1161: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_1163: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1161, 3, 256, 9223372036854775807);  slice_1161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_39: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_19);  add_72 = getitem_19 = None
        add_73: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_38: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_9);  sub_39 = rsqrt_9 = None
        mul_39: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg81_1);  mul_38 = arg81_1 = None
        add_74: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_39, arg82_1);  mul_39 = arg82_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_315: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_74, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_65: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format)
        view_360: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_65, [4096, 768]);  clone_65 = None
        permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_20: "f32[4096, 768]" = torch.ops.aten.mm.default(view_360, permute_316);  view_360 = permute_316 = None
        view_361: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_20, [1024, 4, 768]);  mm_20 = None
        add_75: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_361, arg84_1);  view_361 = arg84_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_50: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_75, 8.0);  add_75 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_372: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_50, [1024, 4, 12, 64]);  div_50 = None
        permute_326: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_372, [1, 0, 2, 3]);  view_372 = None
        permute_327: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_326, [0, 2, 1, 3]);  permute_326 = None
        view_373: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_327, [48, 1024, 64]);  permute_327 = None
        view_374: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_373, [48, 2, 512, 64]);  view_373 = None
        as_strided_32: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_374, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_374 = None
        unsqueeze_97: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
        permute_328: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_97, [0, 1, 2, 4, 3]);  unsqueeze_97 = None
        permute_329: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_328, [0, 1, 2, 4, 3]);  permute_328 = None
        clone_68: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        view_375: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_68, [144, 512, 64]);  clone_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_66: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format)
        view_362: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_66, [4096, 768]);  clone_66 = None
        permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_21: "f32[4096, 768]" = torch.ops.aten.mm.default(view_362, permute_317);  view_362 = permute_317 = None
        view_363: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_21, [1024, 4, 768]);  mm_21 = None
        add_76: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_363, arg86_1);  view_363 = arg86_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_367: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_76, [1024, 4, 12, 64]);  add_76 = None
        permute_320: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_367, [1, 0, 2, 3]);  view_367 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_322: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_320, [0, 2, 1, 3]);  permute_320 = None
        view_369: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_322, [48, 1024, 64]);  permute_322 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_371: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_369, [48, 2, 512, 64]);  view_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_31: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_371, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_96: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_31, 4);  as_strided_31 = None
        permute_324: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_96, [0, 1, 4, 2, 3]);  unsqueeze_96 = None
        permute_330: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_324, [0, 1, 4, 3, 2]);  permute_324 = None
        clone_69: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_376: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_69, [144, 64, 512]);  clone_69 = None
        bmm_10: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_375, view_376);  view_375 = view_376 = None
        view_377: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_10, [48, 3, 512, 1, 512]);  bmm_10 = None
        permute_331: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_377, [0, 1, 2, 4, 3]);  view_377 = None
        view_378: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_331, [48, 3, 512, 512]);  permute_331 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_20: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_378, [0, 0, 0, 1], 0.0);  view_378 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_379: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_20, [48, 3, 512, 513]);  constant_pad_nd_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1158: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_379, 2, 0, 256)
        slice_1159: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1158, 3, 0, 257);  slice_1158 = None
        copy_60: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1163, slice_1159);  slice_1163 = slice_1159 = None
        
        # No stacktrace found for following nodes
        slice_tensor_50: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_scatter_default_110: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_50, copy_60, 3, 256, 9223372036854775807);  slice_tensor_50 = copy_60 = None
        slice_scatter_default_111: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_45, slice_scatter_default_110, 1, 0, -1);  full_45 = slice_scatter_default_110 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_103: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_111, 1, -1);  select_103 = None
        select_102: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_111, 1, -1)
        slice_1179: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_102, 2, 256, 9223372036854775807);  select_102 = None
        select_100: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_379, 1, -1)
        slice_1172: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_100, 1, 256, 9223372036854775807);  select_100 = None
        slice_1173: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1172, 2, 0, 257);  slice_1172 = None
        copy_61: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_1179, slice_1173);  slice_1179 = slice_1173 = None
        
        # No stacktrace found for following nodes
        select_int_20: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_111, 1, -1)
        slice_scatter_default_112: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_20, copy_61, 2, 256, 9223372036854775807);  select_int_20 = copy_61 = None
        select_scatter_default_20: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_111, slice_scatter_default_112, 1, -1);  slice_scatter_default_111 = slice_scatter_default_112 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1198: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_20, 1, 1, 9223372036854775807);  slice_1198 = None
        slice_1194: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_20, 1, 1, 9223372036854775807)
        slice_1196: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1194, 3, 0, 256);  slice_1194 = None
        slice_1187: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_379, 2, -257, -1)
        slice_1188: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1187, 3, 257, 9223372036854775807);  slice_1187 = None
        copy_62: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1196, slice_1188);  slice_1196 = slice_1188 = None
        
        # No stacktrace found for following nodes
        slice_tensor_51: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_20, 1, 1, 9223372036854775807)
        slice_scatter_default_113: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_51, copy_62, 3, 0, 256);  slice_tensor_51 = copy_62 = None
        slice_scatter_default_114: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_20, slice_scatter_default_113, 1, 1, 9223372036854775807);  select_scatter_default_20 = slice_scatter_default_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_108: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_114, 1, 0)
        slice_1214: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_108, 1, 1, 256);  select_108 = slice_1214 = None
        select_107: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_114, 1, 0)
        slice_1211: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_107, 1, 1, 256);  select_107 = None
        slice_1212: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1211, 2, 1, 256);  slice_1211 = None
        select_105: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_379, 1, 0);  view_379 = None
        slice_1205: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_105, 1, 0, 255);  select_105 = None
        slice_1206: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1205, 2, -255, 9223372036854775807);  slice_1205 = None
        copy_63: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_1212, slice_1206);  slice_1212 = slice_1206 = None
        
        # No stacktrace found for following nodes
        select_int_21: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_114, 1, 0)
        slice_tensor_52: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_21, 1, 1, 256)
        slice_scatter_default_115: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_52, copy_63, 2, 1, 256);  slice_tensor_52 = copy_63 = None
        slice_scatter_default_116: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_21, slice_scatter_default_115, 1, 1, 256);  select_int_21 = slice_scatter_default_115 = None
        select_scatter_default_21: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_114, slice_scatter_default_116, 1, 0);  slice_scatter_default_114 = slice_scatter_default_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_384: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_21, [4, 12, 1024, 513])
        permute_336: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        slice_1237: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_336, 1, 0, 256);  slice_1237 = None
        view_383: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_21, [4, 12, 1024, 513])
        permute_335: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        slice_1233: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_335, 1, 0, 256);  permute_335 = None
        slice_1235: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1233, 3, 0, 257);  slice_1233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_20: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_98: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_20, -2);  iota_20 = None
        iota_21: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_99: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
        sub_41: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_98, unsqueeze_99);  unsqueeze_98 = unsqueeze_99 = None
        le_10: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_41, 0);  sub_41 = None
        full_default_50: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40: "f32[256, 257]" = torch.ops.aten.where.self(le_10, full_default_50, full_default_51);  le_10 = full_default_50 = full_default_51 = None
        rev_20: "f32[256, 257]" = torch.ops.prims.rev.default(where_40, [0]);  where_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_100: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_20, 0);  rev_20 = None
        unsqueeze_101: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, 2);  unsqueeze_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_20: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_101, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_25: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_20, torch.bool);  expand_20 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_52: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_382: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_21, [4, 12, 1024, 513]);  select_scatter_default_21 = None
        permute_334: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        slice_1225: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_334, 1, 0, 256);  permute_334 = None
        slice_1227: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1225, 3, 0, 257);  slice_1225 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_41: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_25, full_default_52, slice_1227);  convert_element_type_25 = full_default_52 = slice_1227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_64: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1235, where_41);  slice_1235 = where_41 = None
        
        # No stacktrace found for following nodes
        slice_tensor_53: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_336, 1, 0, 256)
        slice_scatter_default_117: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_53, copy_64, 3, 0, 257);  slice_tensor_53 = copy_64 = None
        slice_scatter_default_118: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_336, slice_scatter_default_117, 1, 0, 256);  permute_336 = slice_scatter_default_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_337: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_118, [0, 2, 1, 3]);  slice_scatter_default_118 = None
        view_385: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_337, [48, 4, 256, 513]);  permute_337 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_389: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_385, [4, 12, 1024, 513])
        permute_341: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None
        slice_1260: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_341, 1, -256, 9223372036854775807);  slice_1260 = None
        view_388: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_385, [4, 12, 1024, 513])
        permute_340: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        slice_1256: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_340, 1, -256, 9223372036854775807);  permute_340 = None
        slice_1258: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1256, 3, -257, 9223372036854775807);  slice_1256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_21: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_101, [1, 3]);  unsqueeze_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_21: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_21, [4, 256, 12, 257]);  rev_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_26: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_21, torch.bool);  expand_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_53: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_387: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_385, [4, 12, 1024, 513]);  view_385 = None
        permute_339: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        slice_1248: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_339, 1, -256, 9223372036854775807);  permute_339 = None
        slice_1250: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1248, 3, -257, 9223372036854775807);  slice_1248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_42: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_26, full_default_53, slice_1250);  convert_element_type_26 = full_default_53 = slice_1250 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_65: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1258, where_42);  slice_1258 = where_42 = None
        
        # No stacktrace found for following nodes
        slice_tensor_54: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_341, 1, -256, 9223372036854775807)
        slice_scatter_default_119: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_54, copy_65, 3, -257, 9223372036854775807);  slice_tensor_54 = copy_65 = None
        slice_scatter_default_120: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_341, slice_scatter_default_119, 1, -256, 9223372036854775807);  permute_341 = slice_scatter_default_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_342: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_120, [0, 2, 1, 3]);  slice_scatter_default_120 = None
        view_390: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_342, [48, 4, 256, 513]);  permute_342 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_410: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_390, [4, 12, 1024, 513]);  view_390 = None
        permute_360: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_50: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1277: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1);  slice_1277 = None
        slice_1273: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1)
        slice_1275: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1273, 3, 256, 9223372036854775807);  slice_1273 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_49: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_344: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_49, [0, 2, 1, 3]);  full_49 = None
        view_392: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_344, [4, 1024, 1]);  permute_344 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_394: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_392, [4, 2, 512, 1]);  view_392 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_33: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_394, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_394 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_104: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_33, 4);  as_strided_33 = None
        permute_346: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_104, [0, 1, 2, 4, 3]);  unsqueeze_104 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_54: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_5: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_102: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_5, 2);  ne_5 = None
        unsqueeze_103: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_27: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_103, torch.float32)
        where_43: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_103, full_default_54, convert_element_type_27);  unsqueeze_103 = full_default_54 = convert_element_type_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_345: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_43, [0, 2, 1, 3]);  where_43 = None
        view_393: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_345, [4, 1024, 1]);  permute_345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_395: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_393, [4, 2, 512, 1]);  view_393 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_34: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_395, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_395 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_105: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_34, 4);  as_strided_34 = None
        permute_347: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_105, [0, 1, 4, 2, 3]);  unsqueeze_105 = None
        mul_40: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_346, permute_347);  permute_346 = permute_347 = None
        view_396: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_40, [4, 3, 512, 512]);  mul_40 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_21: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_396, [0, 0, 0, 1], 0.0);  view_396 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_397: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_21, [4, 3, 512, 513]);  constant_pad_nd_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1270: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_397, 2, 0, 256)
        slice_1271: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1270, 3, 0, 257);  slice_1270 = None
        copy_66: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1275, slice_1271);  slice_1275 = slice_1271 = None
        
        # No stacktrace found for following nodes
        slice_tensor_55: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_50, 1, 0, -1)
        slice_scatter_default_121: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_55, copy_66, 3, 256, 9223372036854775807);  slice_tensor_55 = copy_66 = None
        slice_scatter_default_122: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_50, slice_scatter_default_121, 1, 0, -1);  full_50 = slice_scatter_default_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_113: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_122, 1, -1);  select_113 = None
        select_112: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_122, 1, -1)
        slice_1291: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_112, 2, 256, 9223372036854775807);  select_112 = None
        select_110: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_397, 1, -1)
        slice_1284: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_110, 1, 256, 9223372036854775807);  select_110 = None
        slice_1285: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1284, 2, 0, 257);  slice_1284 = None
        copy_67: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_1291, slice_1285);  slice_1291 = slice_1285 = None
        
        # No stacktrace found for following nodes
        select_int_22: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_122, 1, -1)
        slice_scatter_default_123: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_22, copy_67, 2, 256, 9223372036854775807);  select_int_22 = copy_67 = None
        select_scatter_default_22: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_122, slice_scatter_default_123, 1, -1);  slice_scatter_default_122 = slice_scatter_default_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1310: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_22, 1, 1, 9223372036854775807);  slice_1310 = None
        slice_1306: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_22, 1, 1, 9223372036854775807)
        slice_1308: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1306, 3, 0, 256);  slice_1306 = None
        slice_1299: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_397, 2, -257, -1)
        slice_1300: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1299, 3, 257, 9223372036854775807);  slice_1299 = None
        copy_68: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1308, slice_1300);  slice_1308 = slice_1300 = None
        
        # No stacktrace found for following nodes
        slice_tensor_56: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_22, 1, 1, 9223372036854775807)
        slice_scatter_default_124: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_56, copy_68, 3, 0, 256);  slice_tensor_56 = copy_68 = None
        slice_scatter_default_125: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_22, slice_scatter_default_124, 1, 1, 9223372036854775807);  select_scatter_default_22 = slice_scatter_default_124 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_118: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_125, 1, 0)
        slice_1326: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_118, 1, 1, 256);  select_118 = slice_1326 = None
        select_117: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_125, 1, 0)
        slice_1323: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_117, 1, 1, 256);  select_117 = None
        slice_1324: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1323, 2, 1, 256);  slice_1323 = None
        select_115: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_397, 1, 0);  view_397 = None
        slice_1317: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_115, 1, 0, 255);  select_115 = None
        slice_1318: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1317, 2, -255, 9223372036854775807);  slice_1317 = None
        copy_69: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_1324, slice_1318);  slice_1324 = slice_1318 = None
        
        # No stacktrace found for following nodes
        select_int_23: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_125, 1, 0)
        slice_tensor_57: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_23, 1, 1, 256)
        slice_scatter_default_126: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_57, copy_69, 2, 1, 256);  slice_tensor_57 = copy_69 = None
        slice_scatter_default_127: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_23, slice_scatter_default_126, 1, 1, 256);  select_int_23 = slice_scatter_default_126 = None
        select_scatter_default_23: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_125, slice_scatter_default_127, 1, 0);  slice_scatter_default_125 = slice_scatter_default_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_402: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_23, [4, 1, 1024, 513])
        permute_352: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        slice_1349: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_352, 1, 0, 256);  slice_1349 = None
        view_401: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_23, [4, 1, 1024, 513])
        permute_351: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
        slice_1345: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_351, 1, 0, 256);  permute_351 = None
        slice_1347: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1345, 3, 0, 257);  slice_1345 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_22: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_106: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_22, -2);  iota_22 = None
        iota_23: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_107: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
        sub_43: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_106, unsqueeze_107);  unsqueeze_106 = unsqueeze_107 = None
        le_11: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_43, 0);  sub_43 = None
        full_default_55: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44: "f32[256, 257]" = torch.ops.aten.where.self(le_11, full_default_55, full_default_56);  le_11 = full_default_55 = full_default_56 = None
        rev_22: "f32[256, 257]" = torch.ops.prims.rev.default(where_44, [0]);  where_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_108: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_22, 0);  rev_22 = None
        unsqueeze_109: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 2);  unsqueeze_108 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_22: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_109, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_28: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_22, torch.bool);  expand_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_57: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_400: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_23, [4, 1, 1024, 513]);  select_scatter_default_23 = None
        permute_350: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        slice_1337: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_350, 1, 0, 256);  permute_350 = None
        slice_1339: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1337, 3, 0, 257);  slice_1337 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_45: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_28, full_default_57, slice_1339);  convert_element_type_28 = full_default_57 = slice_1339 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_70: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1347, where_45);  slice_1347 = where_45 = None
        
        # No stacktrace found for following nodes
        slice_tensor_58: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_352, 1, 0, 256)
        slice_scatter_default_128: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_58, copy_70, 3, 0, 257);  slice_tensor_58 = copy_70 = None
        slice_scatter_default_129: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_352, slice_scatter_default_128, 1, 0, 256);  permute_352 = slice_scatter_default_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_353: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_129, [0, 2, 1, 3]);  slice_scatter_default_129 = None
        view_403: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_353, [4, 4, 256, 513]);  permute_353 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_407: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_403, [4, 1, 1024, 513])
        permute_357: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
        slice_1372: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_357, 1, -256, 9223372036854775807);  slice_1372 = None
        view_406: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_403, [4, 1, 1024, 513])
        permute_356: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        slice_1368: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_356, 1, -256, 9223372036854775807);  permute_356 = None
        slice_1370: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1368, 3, -257, 9223372036854775807);  slice_1368 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_23: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_109, [1, 3]);  unsqueeze_109 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_23: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_23, [4, 256, 1, 257]);  rev_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_29: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_23, torch.bool);  expand_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_58: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_405: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_403, [4, 1, 1024, 513]);  view_403 = None
        permute_355: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
        slice_1360: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_355, 1, -256, 9223372036854775807);  permute_355 = None
        slice_1362: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1360, 3, -257, 9223372036854775807);  slice_1360 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_46: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_29, full_default_58, slice_1362);  convert_element_type_29 = full_default_58 = slice_1362 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_71: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1370, where_46);  slice_1370 = where_46 = None
        
        # No stacktrace found for following nodes
        slice_tensor_59: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_357, 1, -256, 9223372036854775807)
        slice_scatter_default_130: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_59, copy_71, 3, -257, 9223372036854775807);  slice_tensor_59 = copy_71 = None
        slice_scatter_default_131: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_357, slice_scatter_default_130, 1, -256, 9223372036854775807);  permute_357 = slice_scatter_default_130 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_358: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_131, [0, 2, 1, 3]);  slice_scatter_default_131 = None
        view_408: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_358, [4, 4, 256, 513]);  permute_358 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_411: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_408, [4, 1, 1024, 513]);  view_408 = None
        permute_361: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
        add_80: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_360, permute_361);  permute_360 = permute_361 = None
        permute_362: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_80, [0, 2, 1, 3]);  add_80 = None
        view_413: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_362, [48, 4, 256, 513]);  permute_362 = None
        view_414: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_413, [4, 12, 1024, 513]);  view_413 = None
        permute_363: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_70: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
        amax_5: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_70, [-1], True)
        sub_44: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_70, amax_5);  clone_70 = amax_5 = None
        exp_5: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
        sum_6: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_57: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_47: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_111, full_default_59, div_57);  unsqueeze_111 = full_default_59 = div_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_365: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_47, [0, 2, 1, 3]);  where_47 = None
        clone_72: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_416: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_72, [48, 4, 256, 513]);  clone_72 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_23: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_416, [0, 257], 0.0);  view_416 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_418: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_23, [48, 4, -1]);  constant_pad_nd_23 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1382: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_418, 2, 0, -256);  view_418 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_419: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1382, [48, 4, 256, 769]);  slice_1382 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1386: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_419, 3, 0, -1);  view_419 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_112: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1386, 4);  slice_1386 = None
        permute_367: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_112, [0, 1, 2, 4, 3]);  unsqueeze_112 = None
        permute_369: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_367, [0, 1, 2, 4, 3]);  permute_367 = None
        view_420: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_369, [192, 256, 768]);  permute_369 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_67: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format);  permute_315 = None
        view_364: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_67, [4096, 768]);  clone_67 = None
        permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_22: "f32[4096, 768]" = torch.ops.aten.mm.default(view_364, permute_318);  view_364 = permute_318 = None
        view_365: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_22, [1024, 4, 768]);  mm_22 = None
        add_77: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_365, arg88_1);  view_365 = arg88_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_415: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_77, [1024, 4, 12, 64]);  add_77 = None
        permute_364: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_415, [1, 0, 2, 3]);  view_415 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_366: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_364, [0, 2, 1, 3]);  permute_364 = None
        view_417: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_366, [48, 1024, 64]);  permute_366 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_22: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_417, [0, 0, 256, 256], -1.0);  view_417 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_35: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_22, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_22 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_113: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
        permute_368: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_113, [0, 1, 4, 3, 2]);  unsqueeze_113 = None
        permute_370: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_368, [0, 1, 4, 3, 2]);  permute_368 = None
        clone_73: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
        view_421: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_73, [192, 768, 64]);  clone_73 = None
        bmm_11: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_420, view_421);  view_420 = view_421 = None
        view_422: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_11, [48, 4, 256, 1, 64]);  bmm_11 = None
        permute_371: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_422, [0, 1, 2, 4, 3]);  view_422 = None
        view_423: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_371, [48, 4, 256, 64]);  permute_371 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_424: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_423, [4, 12, 1024, 64]);  view_423 = None
        permute_372: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_373: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_372, [1, 0, 2, 3]);  permute_372 = None
        clone_74: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_373, memory_format = torch.contiguous_format);  permute_373 = None
        view_425: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_74, [1024, 4, 768]);  clone_74 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_374: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_425, [1, 0, 2]);  view_425 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_75: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        view_426: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_75, [4096, 768]);  clone_75 = None
        permute_375: "f32[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_23: "f32[4096, 768]" = torch.ops.aten.mm.default(view_426, permute_375);  view_426 = permute_375 = None
        view_427: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_23, [4, 1024, 768]);  mm_23 = None
        add_82: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_427, arg90_1);  view_427 = arg90_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_83: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_82, add_74);  add_82 = add_74 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_20: "f32[4, 1024, 1]" = var_mean_10[0]
        getitem_21: "f32[4, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_46: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_21);  add_83 = getitem_21 = None
        add_84: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_41: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_10);  sub_46 = rsqrt_10 = None
        mul_42: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg91_1);  mul_41 = arg91_1 = None
        add_85: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_42, arg92_1);  mul_42 = arg92_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_428: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_85, [4096, 768])
        permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        
        # No stacktrace found for following nodes
        mm_default_13: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_428, permute_376);  view_428 = permute_376 = None
        add_tensor_13: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_13, arg94_1);  mm_default_13 = arg94_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_429: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_13, [4, 1024, 3072]);  add_tensor_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_43: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.5)
        mul_44: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_429, 0.7071067811865476);  view_429 = None
        erf_5: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_86: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_45: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_43, add_86);  mul_43 = add_86 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_430: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_45, [4096, 3072]);  mul_45 = None
        permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        
        # No stacktrace found for following nodes
        mm_default_12: "f32[4096, 768]" = torch.ops.aten.mm.default(view_430, permute_377);  view_430 = permute_377 = None
        add_tensor_12: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg96_1);  mm_default_12 = arg96_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_431: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [4, 1024, 768]);  add_tensor_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_87: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_431, add_85);  view_431 = add_85 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_22: "f32[4, 1024, 1]" = var_mean_11[0]
        getitem_23: "f32[4, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_129: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_130: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
        full_default_69: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_54: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1396: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1);  slice_1396 = None
        slice_1392: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1)
        slice_1394: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1392, 3, 256, 9223372036854775807);  slice_1392 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_47: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_23);  add_87 = getitem_23 = None
        add_88: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_46: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_11);  sub_47 = rsqrt_11 = None
        mul_47: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_46, arg97_1);  mul_46 = arg97_1 = None
        add_89: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_47, arg98_1);  mul_47 = arg98_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_378: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_89, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_78: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format)
        view_432: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_78, [4096, 768]);  clone_78 = None
        permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_24: "f32[4096, 768]" = torch.ops.aten.mm.default(view_432, permute_379);  view_432 = permute_379 = None
        view_433: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_24, [1024, 4, 768]);  mm_24 = None
        add_90: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_433, arg100_1);  view_433 = arg100_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_60: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_90, 8.0);  add_90 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_444: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_60, [1024, 4, 12, 64]);  div_60 = None
        permute_389: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_444, [1, 0, 2, 3]);  view_444 = None
        permute_390: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_389, [0, 2, 1, 3]);  permute_389 = None
        view_445: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_390, [48, 1024, 64]);  permute_390 = None
        view_446: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_445, [48, 2, 512, 64]);  view_445 = None
        as_strided_38: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_446, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_446 = None
        unsqueeze_116: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_38, 4);  as_strided_38 = None
        permute_391: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 2, 4, 3]);  unsqueeze_116 = None
        permute_392: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_391, [0, 1, 2, 4, 3]);  permute_391 = None
        clone_81: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
        view_447: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_81, [144, 512, 64]);  clone_81 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_79: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format)
        view_434: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_79, [4096, 768]);  clone_79 = None
        permute_380: "f32[768, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_25: "f32[4096, 768]" = torch.ops.aten.mm.default(view_434, permute_380);  view_434 = permute_380 = None
        view_435: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_25, [1024, 4, 768]);  mm_25 = None
        add_91: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_435, arg102_1);  view_435 = arg102_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_439: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_91, [1024, 4, 12, 64]);  add_91 = None
        permute_383: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_439, [1, 0, 2, 3]);  view_439 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_385: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_383, [0, 2, 1, 3]);  permute_383 = None
        view_441: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_385, [48, 1024, 64]);  permute_385 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_443: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_441, [48, 2, 512, 64]);  view_441 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_37: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_443, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_443 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_115: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
        permute_387: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_115, [0, 1, 4, 2, 3]);  unsqueeze_115 = None
        permute_393: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_387, [0, 1, 4, 3, 2]);  permute_387 = None
        clone_82: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
        view_448: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_82, [144, 64, 512]);  clone_82 = None
        bmm_12: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_447, view_448);  view_447 = view_448 = None
        view_449: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_12, [48, 3, 512, 1, 512]);  bmm_12 = None
        permute_394: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_449, [0, 1, 2, 4, 3]);  view_449 = None
        view_450: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_394, [48, 3, 512, 512]);  permute_394 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_24: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_450, [0, 0, 0, 1], 0.0);  view_450 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_451: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_24, [48, 3, 512, 513]);  constant_pad_nd_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1389: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_451, 2, 0, 256)
        slice_1390: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1389, 3, 0, 257);  slice_1389 = None
        copy_72: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1394, slice_1390);  slice_1394 = slice_1390 = None
        
        # No stacktrace found for following nodes
        slice_tensor_60: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_54, 1, 0, -1)
        slice_scatter_default_132: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_60, copy_72, 3, 256, 9223372036854775807);  slice_tensor_60 = copy_72 = None
        slice_scatter_default_133: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_54, slice_scatter_default_132, 1, 0, -1);  full_54 = slice_scatter_default_132 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_123: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_133, 1, -1);  select_123 = None
        select_122: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_133, 1, -1)
        slice_1410: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_122, 2, 256, 9223372036854775807);  select_122 = None
        select_120: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_451, 1, -1)
        slice_1403: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_120, 1, 256, 9223372036854775807);  select_120 = None
        slice_1404: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1403, 2, 0, 257);  slice_1403 = None
        copy_73: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_1410, slice_1404);  slice_1410 = slice_1404 = None
        
        # No stacktrace found for following nodes
        select_int_24: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_133, 1, -1)
        slice_scatter_default_134: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_24, copy_73, 2, 256, 9223372036854775807);  select_int_24 = copy_73 = None
        select_scatter_default_24: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_133, slice_scatter_default_134, 1, -1);  slice_scatter_default_133 = slice_scatter_default_134 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1429: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_24, 1, 1, 9223372036854775807);  slice_1429 = None
        slice_1425: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_24, 1, 1, 9223372036854775807)
        slice_1427: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1425, 3, 0, 256);  slice_1425 = None
        slice_1418: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_451, 2, -257, -1)
        slice_1419: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1418, 3, 257, 9223372036854775807);  slice_1418 = None
        copy_74: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1427, slice_1419);  slice_1427 = slice_1419 = None
        
        # No stacktrace found for following nodes
        slice_tensor_61: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_24, 1, 1, 9223372036854775807)
        slice_scatter_default_135: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_61, copy_74, 3, 0, 256);  slice_tensor_61 = copy_74 = None
        slice_scatter_default_136: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_24, slice_scatter_default_135, 1, 1, 9223372036854775807);  select_scatter_default_24 = slice_scatter_default_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_128: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_136, 1, 0)
        slice_1445: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_128, 1, 1, 256);  select_128 = slice_1445 = None
        select_127: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_136, 1, 0)
        slice_1442: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_127, 1, 1, 256);  select_127 = None
        slice_1443: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1442, 2, 1, 256);  slice_1442 = None
        select_125: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_451, 1, 0);  view_451 = None
        slice_1436: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_125, 1, 0, 255);  select_125 = None
        slice_1437: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1436, 2, -255, 9223372036854775807);  slice_1436 = None
        copy_75: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_1443, slice_1437);  slice_1443 = slice_1437 = None
        
        # No stacktrace found for following nodes
        select_int_25: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_136, 1, 0)
        slice_tensor_62: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_25, 1, 1, 256)
        slice_scatter_default_137: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_62, copy_75, 2, 1, 256);  slice_tensor_62 = copy_75 = None
        slice_scatter_default_138: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_25, slice_scatter_default_137, 1, 1, 256);  select_int_25 = slice_scatter_default_137 = None
        select_scatter_default_25: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_136, slice_scatter_default_138, 1, 0);  slice_scatter_default_136 = slice_scatter_default_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_456: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_25, [4, 12, 1024, 513])
        permute_399: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        slice_1468: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_399, 1, 0, 256);  slice_1468 = None
        view_455: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_25, [4, 12, 1024, 513])
        permute_398: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
        slice_1464: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_398, 1, 0, 256);  permute_398 = None
        slice_1466: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1464, 3, 0, 257);  slice_1464 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_24: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_117: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_24, -2);  iota_24 = None
        iota_25: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_118: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
        sub_49: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_117, unsqueeze_118);  unsqueeze_117 = unsqueeze_118 = None
        le_12: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_49, 0);  sub_49 = None
        full_default_60: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48: "f32[256, 257]" = torch.ops.aten.where.self(le_12, full_default_60, full_default_61);  le_12 = full_default_60 = full_default_61 = None
        rev_24: "f32[256, 257]" = torch.ops.prims.rev.default(where_48, [0]);  where_48 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_119: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_24, 0);  rev_24 = None
        unsqueeze_120: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_24: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_120, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_30: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_24, torch.bool);  expand_24 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_62: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_454: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_25, [4, 12, 1024, 513]);  select_scatter_default_25 = None
        permute_397: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        slice_1456: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_397, 1, 0, 256);  permute_397 = None
        slice_1458: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1456, 3, 0, 257);  slice_1456 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_49: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_30, full_default_62, slice_1458);  convert_element_type_30 = full_default_62 = slice_1458 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_76: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1466, where_49);  slice_1466 = where_49 = None
        
        # No stacktrace found for following nodes
        slice_tensor_63: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_399, 1, 0, 256)
        slice_scatter_default_139: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_63, copy_76, 3, 0, 257);  slice_tensor_63 = copy_76 = None
        slice_scatter_default_140: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_399, slice_scatter_default_139, 1, 0, 256);  permute_399 = slice_scatter_default_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_400: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_140, [0, 2, 1, 3]);  slice_scatter_default_140 = None
        view_457: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_400, [48, 4, 256, 513]);  permute_400 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_461: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_457, [4, 12, 1024, 513])
        permute_404: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_461, [0, 2, 1, 3]);  view_461 = None
        slice_1491: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_404, 1, -256, 9223372036854775807);  slice_1491 = None
        view_460: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_457, [4, 12, 1024, 513])
        permute_403: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        slice_1487: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_403, 1, -256, 9223372036854775807);  permute_403 = None
        slice_1489: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1487, 3, -257, 9223372036854775807);  slice_1487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_25: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_120, [1, 3]);  unsqueeze_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_25: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_25, [4, 256, 12, 257]);  rev_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_31: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_25, torch.bool);  expand_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_63: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_459: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_457, [4, 12, 1024, 513]);  view_457 = None
        permute_402: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        slice_1479: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_402, 1, -256, 9223372036854775807);  permute_402 = None
        slice_1481: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1479, 3, -257, 9223372036854775807);  slice_1479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_50: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_31, full_default_63, slice_1481);  convert_element_type_31 = full_default_63 = slice_1481 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_77: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1489, where_50);  slice_1489 = where_50 = None
        
        # No stacktrace found for following nodes
        slice_tensor_64: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_404, 1, -256, 9223372036854775807)
        slice_scatter_default_141: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_64, copy_77, 3, -257, 9223372036854775807);  slice_tensor_64 = copy_77 = None
        slice_scatter_default_142: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_404, slice_scatter_default_141, 1, -256, 9223372036854775807);  permute_404 = slice_scatter_default_141 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_405: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_142, [0, 2, 1, 3]);  slice_scatter_default_142 = None
        view_462: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_405, [48, 4, 256, 513]);  permute_405 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_482: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_462, [4, 12, 1024, 513]);  view_462 = None
        permute_423: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_59: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1508: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1);  slice_1508 = None
        slice_1504: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1)
        slice_1506: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1504, 3, 256, 9223372036854775807);  slice_1504 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_58: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_407: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_58, [0, 2, 1, 3]);  full_58 = None
        view_464: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_407, [4, 1024, 1]);  permute_407 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_466: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_464, [4, 2, 512, 1]);  view_464 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_39: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_466, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_466 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_123: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_39, 4);  as_strided_39 = None
        permute_409: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_123, [0, 1, 2, 4, 3]);  unsqueeze_123 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_64: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_6: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_121: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_6, 2);  ne_6 = None
        unsqueeze_122: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 3);  unsqueeze_121 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_32: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_122, torch.float32)
        where_51: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_122, full_default_64, convert_element_type_32);  unsqueeze_122 = full_default_64 = convert_element_type_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_408: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_51, [0, 2, 1, 3]);  where_51 = None
        view_465: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_408, [4, 1024, 1]);  permute_408 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_467: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_465, [4, 2, 512, 1]);  view_465 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_40: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_467, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_467 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_124: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_40, 4);  as_strided_40 = None
        permute_410: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_124, [0, 1, 4, 2, 3]);  unsqueeze_124 = None
        mul_48: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_409, permute_410);  permute_409 = permute_410 = None
        view_468: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_48, [4, 3, 512, 512]);  mul_48 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_25: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_468, [0, 0, 0, 1], 0.0);  view_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_469: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_25, [4, 3, 512, 513]);  constant_pad_nd_25 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1501: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_469, 2, 0, 256)
        slice_1502: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1501, 3, 0, 257);  slice_1501 = None
        copy_78: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1506, slice_1502);  slice_1506 = slice_1502 = None
        
        # No stacktrace found for following nodes
        slice_tensor_65: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_59, 1, 0, -1)
        slice_scatter_default_143: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_65, copy_78, 3, 256, 9223372036854775807);  slice_tensor_65 = copy_78 = None
        slice_scatter_default_144: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_59, slice_scatter_default_143, 1, 0, -1);  full_59 = slice_scatter_default_143 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_133: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_144, 1, -1);  select_133 = None
        select_132: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_144, 1, -1)
        slice_1522: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_132, 2, 256, 9223372036854775807);  select_132 = None
        select_130: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_469, 1, -1)
        slice_1515: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_130, 1, 256, 9223372036854775807);  select_130 = None
        slice_1516: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1515, 2, 0, 257);  slice_1515 = None
        copy_79: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_1522, slice_1516);  slice_1522 = slice_1516 = None
        
        # No stacktrace found for following nodes
        select_int_26: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_144, 1, -1)
        slice_scatter_default_145: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_26, copy_79, 2, 256, 9223372036854775807);  select_int_26 = copy_79 = None
        select_scatter_default_26: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_144, slice_scatter_default_145, 1, -1);  slice_scatter_default_144 = slice_scatter_default_145 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1541: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_26, 1, 1, 9223372036854775807);  slice_1541 = None
        slice_1537: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_26, 1, 1, 9223372036854775807)
        slice_1539: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1537, 3, 0, 256);  slice_1537 = None
        slice_1530: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_469, 2, -257, -1)
        slice_1531: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1530, 3, 257, 9223372036854775807);  slice_1530 = None
        copy_80: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1539, slice_1531);  slice_1539 = slice_1531 = None
        
        # No stacktrace found for following nodes
        slice_tensor_66: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_26, 1, 1, 9223372036854775807)
        slice_scatter_default_146: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_66, copy_80, 3, 0, 256);  slice_tensor_66 = copy_80 = None
        slice_scatter_default_147: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_26, slice_scatter_default_146, 1, 1, 9223372036854775807);  select_scatter_default_26 = slice_scatter_default_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_138: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_147, 1, 0)
        slice_1557: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_138, 1, 1, 256);  select_138 = slice_1557 = None
        select_137: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_147, 1, 0)
        slice_1554: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_137, 1, 1, 256);  select_137 = None
        slice_1555: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1554, 2, 1, 256);  slice_1554 = None
        select_135: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_469, 1, 0);  view_469 = None
        slice_1548: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_135, 1, 0, 255);  select_135 = None
        slice_1549: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1548, 2, -255, 9223372036854775807);  slice_1548 = None
        copy_81: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_1555, slice_1549);  slice_1555 = slice_1549 = None
        
        # No stacktrace found for following nodes
        select_int_27: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_147, 1, 0)
        slice_tensor_67: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_27, 1, 1, 256)
        slice_scatter_default_148: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_67, copy_81, 2, 1, 256);  slice_tensor_67 = copy_81 = None
        slice_scatter_default_149: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_27, slice_scatter_default_148, 1, 1, 256);  select_int_27 = slice_scatter_default_148 = None
        select_scatter_default_27: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_147, slice_scatter_default_149, 1, 0);  slice_scatter_default_147 = slice_scatter_default_149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_474: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_27, [4, 1, 1024, 513])
        permute_415: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None
        slice_1580: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_415, 1, 0, 256);  slice_1580 = None
        view_473: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_27, [4, 1, 1024, 513])
        permute_414: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None
        slice_1576: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_414, 1, 0, 256);  permute_414 = None
        slice_1578: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1576, 3, 0, 257);  slice_1576 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_26: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_125: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_26, -2);  iota_26 = None
        iota_27: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_126: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
        sub_51: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_125, unsqueeze_126);  unsqueeze_125 = unsqueeze_126 = None
        le_13: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_51, 0);  sub_51 = None
        full_default_65: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_52: "f32[256, 257]" = torch.ops.aten.where.self(le_13, full_default_65, full_default_66);  le_13 = full_default_65 = full_default_66 = None
        rev_26: "f32[256, 257]" = torch.ops.prims.rev.default(where_52, [0]);  where_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_127: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_26, 0);  rev_26 = None
        unsqueeze_128: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_26: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_128, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_33: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_26, torch.bool);  expand_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_67: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_472: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_27, [4, 1, 1024, 513]);  select_scatter_default_27 = None
        permute_413: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        slice_1568: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_413, 1, 0, 256);  permute_413 = None
        slice_1570: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1568, 3, 0, 257);  slice_1568 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_53: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_33, full_default_67, slice_1570);  convert_element_type_33 = full_default_67 = slice_1570 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_82: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1578, where_53);  slice_1578 = where_53 = None
        
        # No stacktrace found for following nodes
        slice_tensor_68: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_415, 1, 0, 256)
        slice_scatter_default_150: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_68, copy_82, 3, 0, 257);  slice_tensor_68 = copy_82 = None
        slice_scatter_default_151: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_415, slice_scatter_default_150, 1, 0, 256);  permute_415 = slice_scatter_default_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_416: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_151, [0, 2, 1, 3]);  slice_scatter_default_151 = None
        view_475: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_416, [4, 4, 256, 513]);  permute_416 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_479: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_475, [4, 1, 1024, 513])
        permute_420: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
        slice_1603: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_420, 1, -256, 9223372036854775807);  slice_1603 = None
        view_478: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_475, [4, 1, 1024, 513])
        permute_419: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        slice_1599: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_419, 1, -256, 9223372036854775807);  permute_419 = None
        slice_1601: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1599, 3, -257, 9223372036854775807);  slice_1599 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_27: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_128, [1, 3]);  unsqueeze_128 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_27: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_27, [4, 256, 1, 257]);  rev_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_34: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_27, torch.bool);  expand_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_68: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_477: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_475, [4, 1, 1024, 513]);  view_475 = None
        permute_418: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
        slice_1591: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_418, 1, -256, 9223372036854775807);  permute_418 = None
        slice_1593: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1591, 3, -257, 9223372036854775807);  slice_1591 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_54: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_34, full_default_68, slice_1593);  convert_element_type_34 = full_default_68 = slice_1593 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_83: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1601, where_54);  slice_1601 = where_54 = None
        
        # No stacktrace found for following nodes
        slice_tensor_69: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_420, 1, -256, 9223372036854775807)
        slice_scatter_default_152: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_69, copy_83, 3, -257, 9223372036854775807);  slice_tensor_69 = copy_83 = None
        slice_scatter_default_153: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_420, slice_scatter_default_152, 1, -256, 9223372036854775807);  permute_420 = slice_scatter_default_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_421: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_153, [0, 2, 1, 3]);  slice_scatter_default_153 = None
        view_480: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_421, [4, 4, 256, 513]);  permute_421 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_483: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_480, [4, 1, 1024, 513]);  view_480 = None
        permute_424: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_483, [0, 2, 1, 3]);  view_483 = None
        add_95: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_423, permute_424);  permute_423 = permute_424 = None
        permute_425: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_95, [0, 2, 1, 3]);  add_95 = None
        view_485: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_425, [48, 4, 256, 513]);  permute_425 = None
        view_486: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_485, [4, 12, 1024, 513]);  view_485 = None
        permute_426: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_83: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
        amax_6: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_83, [-1], True)
        sub_52: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_83, amax_6);  clone_83 = amax_6 = None
        exp_6: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_7: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_67: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_55: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_130, full_default_69, div_67);  unsqueeze_130 = full_default_69 = div_67 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_428: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_55, [0, 2, 1, 3]);  where_55 = None
        clone_85: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_428, memory_format = torch.contiguous_format);  permute_428 = None
        view_488: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_85, [48, 4, 256, 513]);  clone_85 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_27: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_488, [0, 257], 0.0);  view_488 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_490: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_27, [48, 4, -1]);  constant_pad_nd_27 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1613: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_490, 2, 0, -256);  view_490 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_491: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1613, [48, 4, 256, 769]);  slice_1613 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1617: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_491, 3, 0, -1);  view_491 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_131: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1617, 4);  slice_1617 = None
        permute_430: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_131, [0, 1, 2, 4, 3]);  unsqueeze_131 = None
        permute_432: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_430, [0, 1, 2, 4, 3]);  permute_430 = None
        view_492: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_432, [192, 256, 768]);  permute_432 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_80: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format);  permute_378 = None
        view_436: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_80, [4096, 768]);  clone_80 = None
        permute_381: "f32[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_26: "f32[4096, 768]" = torch.ops.aten.mm.default(view_436, permute_381);  view_436 = permute_381 = None
        view_437: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_26, [1024, 4, 768]);  mm_26 = None
        add_92: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_437, arg104_1);  view_437 = arg104_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_487: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_92, [1024, 4, 12, 64]);  add_92 = None
        permute_427: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_487, [1, 0, 2, 3]);  view_487 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_429: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_427, [0, 2, 1, 3]);  permute_427 = None
        view_489: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_429, [48, 1024, 64]);  permute_429 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_26: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_489, [0, 0, 256, 256], -1.0);  view_489 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_41: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_26, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_26 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_132: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_41, 4);  as_strided_41 = None
        permute_431: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_132, [0, 1, 4, 3, 2]);  unsqueeze_132 = None
        permute_433: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_431, [0, 1, 4, 3, 2]);  permute_431 = None
        clone_86: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
        view_493: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_86, [192, 768, 64]);  clone_86 = None
        bmm_13: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_492, view_493);  view_492 = view_493 = None
        view_494: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_13, [48, 4, 256, 1, 64]);  bmm_13 = None
        permute_434: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_494, [0, 1, 2, 4, 3]);  view_494 = None
        view_495: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_434, [48, 4, 256, 64]);  permute_434 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_496: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_495, [4, 12, 1024, 64]);  view_495 = None
        permute_435: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_436: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_435, [1, 0, 2, 3]);  permute_435 = None
        clone_87: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
        view_497: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_87, [1024, 4, 768]);  clone_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_437: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_497, [1, 0, 2]);  view_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_88: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
        view_498: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_88, [4096, 768]);  clone_88 = None
        permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_27: "f32[4096, 768]" = torch.ops.aten.mm.default(view_498, permute_438);  view_498 = permute_438 = None
        view_499: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_27, [4, 1024, 768]);  mm_27 = None
        add_97: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_499, arg106_1);  view_499 = arg106_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_98: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_97, add_89);  add_97 = add_89 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_24: "f32[4, 1024, 1]" = var_mean_12[0]
        getitem_25: "f32[4, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_54: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_98, getitem_25);  add_98 = getitem_25 = None
        add_99: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_49: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_12);  sub_54 = rsqrt_12 = None
        mul_50: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg107_1);  mul_49 = arg107_1 = None
        add_100: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_50, arg108_1);  mul_50 = arg108_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_500: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_100, [4096, 768])
        permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        
        # No stacktrace found for following nodes
        mm_default_11: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_500, permute_439);  view_500 = permute_439 = None
        add_tensor_11: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_11, arg110_1);  mm_default_11 = arg110_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_501: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_11, [4, 1024, 3072]);  add_tensor_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_51: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        mul_52: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_501, 0.7071067811865476);  view_501 = None
        erf_6: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_101: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_53: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_101);  mul_51 = add_101 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_502: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_53, [4096, 3072]);  mul_53 = None
        permute_440: "f32[3072, 768]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        
        # No stacktrace found for following nodes
        mm_default_10: "f32[4096, 768]" = torch.ops.aten.mm.default(view_502, permute_440);  view_502 = permute_440 = None
        add_tensor_10: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_10, arg112_1);  mm_default_10 = arg112_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_10, [4, 1024, 768]);  add_tensor_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_102: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_503, add_100);  view_503 = add_100 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
        getitem_26: "f32[4, 1024, 1]" = var_mean_13[0]
        getitem_27: "f32[4, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_148: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_149: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, 3);  unsqueeze_148 = None
        full_default_79: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_63: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1627: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1);  slice_1627 = None
        slice_1623: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1)
        slice_1625: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1623, 3, 256, 9223372036854775807);  slice_1623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_55: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_102, getitem_27);  add_102 = getitem_27 = None
        add_103: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
        mul_54: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_13);  sub_55 = rsqrt_13 = None
        mul_55: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg113_1);  mul_54 = arg113_1 = None
        add_104: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_55, arg114_1);  mul_55 = arg114_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_441: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_104, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_91: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format)
        view_504: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_91, [4096, 768]);  clone_91 = None
        permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_28: "f32[4096, 768]" = torch.ops.aten.mm.default(view_504, permute_442);  view_504 = permute_442 = None
        view_505: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_28, [1024, 4, 768]);  mm_28 = None
        add_105: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_505, arg116_1);  view_505 = arg116_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_70: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_105, 8.0);  add_105 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_516: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_70, [1024, 4, 12, 64]);  div_70 = None
        permute_452: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_516, [1, 0, 2, 3]);  view_516 = None
        permute_453: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_452, [0, 2, 1, 3]);  permute_452 = None
        view_517: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_453, [48, 1024, 64]);  permute_453 = None
        view_518: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_517, [48, 2, 512, 64]);  view_517 = None
        as_strided_44: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_518, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_518 = None
        unsqueeze_135: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_44, 4);  as_strided_44 = None
        permute_454: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_135, [0, 1, 2, 4, 3]);  unsqueeze_135 = None
        permute_455: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_454, [0, 1, 2, 4, 3]);  permute_454 = None
        clone_94: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
        view_519: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_94, [144, 512, 64]);  clone_94 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_92: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format)
        view_506: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_92, [4096, 768]);  clone_92 = None
        permute_443: "f32[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_29: "f32[4096, 768]" = torch.ops.aten.mm.default(view_506, permute_443);  view_506 = permute_443 = None
        view_507: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_29, [1024, 4, 768]);  mm_29 = None
        add_106: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_507, arg118_1);  view_507 = arg118_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_511: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_106, [1024, 4, 12, 64]);  add_106 = None
        permute_446: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_511, [1, 0, 2, 3]);  view_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_448: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_446, [0, 2, 1, 3]);  permute_446 = None
        view_513: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_448, [48, 1024, 64]);  permute_448 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_515: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_513, [48, 2, 512, 64]);  view_513 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_43: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_515, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_515 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_134: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_43, 4);  as_strided_43 = None
        permute_450: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_134, [0, 1, 4, 2, 3]);  unsqueeze_134 = None
        permute_456: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_450, [0, 1, 4, 3, 2]);  permute_450 = None
        clone_95: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_456, memory_format = torch.contiguous_format);  permute_456 = None
        view_520: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_95, [144, 64, 512]);  clone_95 = None
        bmm_14: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_519, view_520);  view_519 = view_520 = None
        view_521: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_14, [48, 3, 512, 1, 512]);  bmm_14 = None
        permute_457: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_521, [0, 1, 2, 4, 3]);  view_521 = None
        view_522: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_457, [48, 3, 512, 512]);  permute_457 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_28: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_522, [0, 0, 0, 1], 0.0);  view_522 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_523: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_28, [48, 3, 512, 513]);  constant_pad_nd_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1620: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_523, 2, 0, 256)
        slice_1621: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1620, 3, 0, 257);  slice_1620 = None
        copy_84: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1625, slice_1621);  slice_1625 = slice_1621 = None
        
        # No stacktrace found for following nodes
        slice_tensor_70: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_63, 1, 0, -1)
        slice_scatter_default_154: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_70, copy_84, 3, 256, 9223372036854775807);  slice_tensor_70 = copy_84 = None
        slice_scatter_default_155: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_63, slice_scatter_default_154, 1, 0, -1);  full_63 = slice_scatter_default_154 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_143: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_155, 1, -1);  select_143 = None
        select_142: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_155, 1, -1)
        slice_1641: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_142, 2, 256, 9223372036854775807);  select_142 = None
        select_140: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_523, 1, -1)
        slice_1634: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_140, 1, 256, 9223372036854775807);  select_140 = None
        slice_1635: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1634, 2, 0, 257);  slice_1634 = None
        copy_85: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_1641, slice_1635);  slice_1641 = slice_1635 = None
        
        # No stacktrace found for following nodes
        select_int_28: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_155, 1, -1)
        slice_scatter_default_156: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_28, copy_85, 2, 256, 9223372036854775807);  select_int_28 = copy_85 = None
        select_scatter_default_28: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_155, slice_scatter_default_156, 1, -1);  slice_scatter_default_155 = slice_scatter_default_156 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1660: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_28, 1, 1, 9223372036854775807);  slice_1660 = None
        slice_1656: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_28, 1, 1, 9223372036854775807)
        slice_1658: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1656, 3, 0, 256);  slice_1656 = None
        slice_1649: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_523, 2, -257, -1)
        slice_1650: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1649, 3, 257, 9223372036854775807);  slice_1649 = None
        copy_86: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1658, slice_1650);  slice_1658 = slice_1650 = None
        
        # No stacktrace found for following nodes
        slice_tensor_71: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_28, 1, 1, 9223372036854775807)
        slice_scatter_default_157: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_71, copy_86, 3, 0, 256);  slice_tensor_71 = copy_86 = None
        slice_scatter_default_158: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_28, slice_scatter_default_157, 1, 1, 9223372036854775807);  select_scatter_default_28 = slice_scatter_default_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_148: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_158, 1, 0)
        slice_1676: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_148, 1, 1, 256);  select_148 = slice_1676 = None
        select_147: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_158, 1, 0)
        slice_1673: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_147, 1, 1, 256);  select_147 = None
        slice_1674: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1673, 2, 1, 256);  slice_1673 = None
        select_145: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_523, 1, 0);  view_523 = None
        slice_1667: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_145, 1, 0, 255);  select_145 = None
        slice_1668: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1667, 2, -255, 9223372036854775807);  slice_1667 = None
        copy_87: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_1674, slice_1668);  slice_1674 = slice_1668 = None
        
        # No stacktrace found for following nodes
        select_int_29: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_158, 1, 0)
        slice_tensor_72: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_29, 1, 1, 256)
        slice_scatter_default_159: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_72, copy_87, 2, 1, 256);  slice_tensor_72 = copy_87 = None
        slice_scatter_default_160: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_29, slice_scatter_default_159, 1, 1, 256);  select_int_29 = slice_scatter_default_159 = None
        select_scatter_default_29: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_158, slice_scatter_default_160, 1, 0);  slice_scatter_default_158 = slice_scatter_default_160 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_528: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_29, [4, 12, 1024, 513])
        permute_462: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
        slice_1699: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_462, 1, 0, 256);  slice_1699 = None
        view_527: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_29, [4, 12, 1024, 513])
        permute_461: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
        slice_1695: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_461, 1, 0, 256);  permute_461 = None
        slice_1697: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1695, 3, 0, 257);  slice_1695 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_28: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_136: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_28, -2);  iota_28 = None
        iota_29: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_137: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
        sub_57: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_136, unsqueeze_137);  unsqueeze_136 = unsqueeze_137 = None
        le_14: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_57, 0);  sub_57 = None
        full_default_70: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_56: "f32[256, 257]" = torch.ops.aten.where.self(le_14, full_default_70, full_default_71);  le_14 = full_default_70 = full_default_71 = None
        rev_28: "f32[256, 257]" = torch.ops.prims.rev.default(where_56, [0]);  where_56 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_138: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_28, 0);  rev_28 = None
        unsqueeze_139: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 2);  unsqueeze_138 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_28: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_139, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_35: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_28, torch.bool);  expand_28 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_72: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_526: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_29, [4, 12, 1024, 513]);  select_scatter_default_29 = None
        permute_460: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
        slice_1687: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_460, 1, 0, 256);  permute_460 = None
        slice_1689: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1687, 3, 0, 257);  slice_1687 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_57: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_35, full_default_72, slice_1689);  convert_element_type_35 = full_default_72 = slice_1689 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_88: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1697, where_57);  slice_1697 = where_57 = None
        
        # No stacktrace found for following nodes
        slice_tensor_73: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_462, 1, 0, 256)
        slice_scatter_default_161: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_73, copy_88, 3, 0, 257);  slice_tensor_73 = copy_88 = None
        slice_scatter_default_162: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_462, slice_scatter_default_161, 1, 0, 256);  permute_462 = slice_scatter_default_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_463: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_162, [0, 2, 1, 3]);  slice_scatter_default_162 = None
        view_529: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_463, [48, 4, 256, 513]);  permute_463 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_533: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_529, [4, 12, 1024, 513])
        permute_467: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
        slice_1722: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_467, 1, -256, 9223372036854775807);  slice_1722 = None
        view_532: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_529, [4, 12, 1024, 513])
        permute_466: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        slice_1718: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_466, 1, -256, 9223372036854775807);  permute_466 = None
        slice_1720: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1718, 3, -257, 9223372036854775807);  slice_1718 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_29: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_139, [1, 3]);  unsqueeze_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_29: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_29, [4, 256, 12, 257]);  rev_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_36: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_29, torch.bool);  expand_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_73: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_531: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_529, [4, 12, 1024, 513]);  view_529 = None
        permute_465: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
        slice_1710: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_465, 1, -256, 9223372036854775807);  permute_465 = None
        slice_1712: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1710, 3, -257, 9223372036854775807);  slice_1710 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_58: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_36, full_default_73, slice_1712);  convert_element_type_36 = full_default_73 = slice_1712 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_89: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1720, where_58);  slice_1720 = where_58 = None
        
        # No stacktrace found for following nodes
        slice_tensor_74: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_467, 1, -256, 9223372036854775807)
        slice_scatter_default_163: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_74, copy_89, 3, -257, 9223372036854775807);  slice_tensor_74 = copy_89 = None
        slice_scatter_default_164: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_467, slice_scatter_default_163, 1, -256, 9223372036854775807);  permute_467 = slice_scatter_default_163 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_468: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_164, [0, 2, 1, 3]);  slice_scatter_default_164 = None
        view_534: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_468, [48, 4, 256, 513]);  permute_468 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_554: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_534, [4, 12, 1024, 513]);  view_534 = None
        permute_486: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_68: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1739: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1);  slice_1739 = None
        slice_1735: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1)
        slice_1737: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1735, 3, 256, 9223372036854775807);  slice_1735 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_67: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_470: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_67, [0, 2, 1, 3]);  full_67 = None
        view_536: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_470, [4, 1024, 1]);  permute_470 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_538: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_536, [4, 2, 512, 1]);  view_536 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_45: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_538, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_538 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_142: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_45, 4);  as_strided_45 = None
        permute_472: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_142, [0, 1, 2, 4, 3]);  unsqueeze_142 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_74: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_7: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_140: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_7, 2);  ne_7 = None
        unsqueeze_141: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_37: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_141, torch.float32)
        where_59: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_141, full_default_74, convert_element_type_37);  unsqueeze_141 = full_default_74 = convert_element_type_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_471: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_59, [0, 2, 1, 3]);  where_59 = None
        view_537: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_471, [4, 1024, 1]);  permute_471 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_539: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_537, [4, 2, 512, 1]);  view_537 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_46: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_539, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_539 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_143: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_46, 4);  as_strided_46 = None
        permute_473: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_143, [0, 1, 4, 2, 3]);  unsqueeze_143 = None
        mul_56: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_472, permute_473);  permute_472 = permute_473 = None
        view_540: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_56, [4, 3, 512, 512]);  mul_56 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_29: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_540, [0, 0, 0, 1], 0.0);  view_540 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_541: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_29, [4, 3, 512, 513]);  constant_pad_nd_29 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1732: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_541, 2, 0, 256)
        slice_1733: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1732, 3, 0, 257);  slice_1732 = None
        copy_90: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1737, slice_1733);  slice_1737 = slice_1733 = None
        
        # No stacktrace found for following nodes
        slice_tensor_75: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_68, 1, 0, -1)
        slice_scatter_default_165: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_75, copy_90, 3, 256, 9223372036854775807);  slice_tensor_75 = copy_90 = None
        slice_scatter_default_166: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_68, slice_scatter_default_165, 1, 0, -1);  full_68 = slice_scatter_default_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_153: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_166, 1, -1);  select_153 = None
        select_152: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_166, 1, -1)
        slice_1753: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_152, 2, 256, 9223372036854775807);  select_152 = None
        select_150: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_541, 1, -1)
        slice_1746: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_150, 1, 256, 9223372036854775807);  select_150 = None
        slice_1747: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1746, 2, 0, 257);  slice_1746 = None
        copy_91: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_1753, slice_1747);  slice_1753 = slice_1747 = None
        
        # No stacktrace found for following nodes
        select_int_30: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_166, 1, -1)
        slice_scatter_default_167: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_30, copy_91, 2, 256, 9223372036854775807);  select_int_30 = copy_91 = None
        select_scatter_default_30: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_166, slice_scatter_default_167, 1, -1);  slice_scatter_default_166 = slice_scatter_default_167 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1772: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_30, 1, 1, 9223372036854775807);  slice_1772 = None
        slice_1768: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_30, 1, 1, 9223372036854775807)
        slice_1770: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1768, 3, 0, 256);  slice_1768 = None
        slice_1761: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_541, 2, -257, -1)
        slice_1762: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1761, 3, 257, 9223372036854775807);  slice_1761 = None
        copy_92: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1770, slice_1762);  slice_1770 = slice_1762 = None
        
        # No stacktrace found for following nodes
        slice_tensor_76: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_30, 1, 1, 9223372036854775807)
        slice_scatter_default_168: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_76, copy_92, 3, 0, 256);  slice_tensor_76 = copy_92 = None
        slice_scatter_default_169: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_30, slice_scatter_default_168, 1, 1, 9223372036854775807);  select_scatter_default_30 = slice_scatter_default_168 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_158: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_169, 1, 0)
        slice_1788: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_158, 1, 1, 256);  select_158 = slice_1788 = None
        select_157: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_169, 1, 0)
        slice_1785: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_157, 1, 1, 256);  select_157 = None
        slice_1786: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1785, 2, 1, 256);  slice_1785 = None
        select_155: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_541, 1, 0);  view_541 = None
        slice_1779: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_155, 1, 0, 255);  select_155 = None
        slice_1780: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1779, 2, -255, 9223372036854775807);  slice_1779 = None
        copy_93: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_1786, slice_1780);  slice_1786 = slice_1780 = None
        
        # No stacktrace found for following nodes
        select_int_31: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_169, 1, 0)
        slice_tensor_77: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_31, 1, 1, 256)
        slice_scatter_default_170: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_77, copy_93, 2, 1, 256);  slice_tensor_77 = copy_93 = None
        slice_scatter_default_171: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_31, slice_scatter_default_170, 1, 1, 256);  select_int_31 = slice_scatter_default_170 = None
        select_scatter_default_31: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_169, slice_scatter_default_171, 1, 0);  slice_scatter_default_169 = slice_scatter_default_171 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_546: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_31, [4, 1, 1024, 513])
        permute_478: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
        slice_1811: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_478, 1, 0, 256);  slice_1811 = None
        view_545: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_31, [4, 1, 1024, 513])
        permute_477: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
        slice_1807: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_477, 1, 0, 256);  permute_477 = None
        slice_1809: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1807, 3, 0, 257);  slice_1807 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_30: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_144: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_30, -2);  iota_30 = None
        iota_31: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_145: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
        sub_59: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_144, unsqueeze_145);  unsqueeze_144 = unsqueeze_145 = None
        le_15: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_59, 0);  sub_59 = None
        full_default_75: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_60: "f32[256, 257]" = torch.ops.aten.where.self(le_15, full_default_75, full_default_76);  le_15 = full_default_75 = full_default_76 = None
        rev_30: "f32[256, 257]" = torch.ops.prims.rev.default(where_60, [0]);  where_60 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_146: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_30, 0);  rev_30 = None
        unsqueeze_147: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_30: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_147, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_38: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_30, torch.bool);  expand_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_77: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_544: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_31, [4, 1, 1024, 513]);  select_scatter_default_31 = None
        permute_476: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
        slice_1799: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_476, 1, 0, 256);  permute_476 = None
        slice_1801: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1799, 3, 0, 257);  slice_1799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_61: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_38, full_default_77, slice_1801);  convert_element_type_38 = full_default_77 = slice_1801 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_94: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1809, where_61);  slice_1809 = where_61 = None
        
        # No stacktrace found for following nodes
        slice_tensor_78: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_478, 1, 0, 256)
        slice_scatter_default_172: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_78, copy_94, 3, 0, 257);  slice_tensor_78 = copy_94 = None
        slice_scatter_default_173: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_478, slice_scatter_default_172, 1, 0, 256);  permute_478 = slice_scatter_default_172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_479: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_173, [0, 2, 1, 3]);  slice_scatter_default_173 = None
        view_547: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_479, [4, 4, 256, 513]);  permute_479 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_551: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_547, [4, 1, 1024, 513])
        permute_483: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_551, [0, 2, 1, 3]);  view_551 = None
        slice_1834: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_483, 1, -256, 9223372036854775807);  slice_1834 = None
        view_550: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_547, [4, 1, 1024, 513])
        permute_482: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
        slice_1830: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_482, 1, -256, 9223372036854775807);  permute_482 = None
        slice_1832: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1830, 3, -257, 9223372036854775807);  slice_1830 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_31: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_147, [1, 3]);  unsqueeze_147 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_31: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_31, [4, 256, 1, 257]);  rev_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_39: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_31, torch.bool);  expand_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_78: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_549: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_547, [4, 1, 1024, 513]);  view_547 = None
        permute_481: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_549, [0, 2, 1, 3]);  view_549 = None
        slice_1822: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_481, 1, -256, 9223372036854775807);  permute_481 = None
        slice_1824: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_1822, 3, -257, 9223372036854775807);  slice_1822 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_62: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_39, full_default_78, slice_1824);  convert_element_type_39 = full_default_78 = slice_1824 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_95: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_1832, where_62);  slice_1832 = where_62 = None
        
        # No stacktrace found for following nodes
        slice_tensor_79: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_483, 1, -256, 9223372036854775807)
        slice_scatter_default_174: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_79, copy_95, 3, -257, 9223372036854775807);  slice_tensor_79 = copy_95 = None
        slice_scatter_default_175: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_483, slice_scatter_default_174, 1, -256, 9223372036854775807);  permute_483 = slice_scatter_default_174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_484: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_175, [0, 2, 1, 3]);  slice_scatter_default_175 = None
        view_552: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_484, [4, 4, 256, 513]);  permute_484 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_555: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_552, [4, 1, 1024, 513]);  view_552 = None
        permute_487: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
        add_110: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_486, permute_487);  permute_486 = permute_487 = None
        permute_488: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_110, [0, 2, 1, 3]);  add_110 = None
        view_557: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_488, [48, 4, 256, 513]);  permute_488 = None
        view_558: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_557, [4, 12, 1024, 513]);  view_557 = None
        permute_489: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_96: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
        amax_7: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_96, [-1], True)
        sub_60: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_96, amax_7);  clone_96 = amax_7 = None
        exp_7: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_8: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_77: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_63: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_149, full_default_79, div_77);  unsqueeze_149 = full_default_79 = div_77 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_491: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_63, [0, 2, 1, 3]);  where_63 = None
        clone_98: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
        view_560: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_98, [48, 4, 256, 513]);  clone_98 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_31: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_560, [0, 257], 0.0);  view_560 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_562: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_31, [48, 4, -1]);  constant_pad_nd_31 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_1844: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_562, 2, 0, -256);  view_562 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_563: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_1844, [48, 4, 256, 769]);  slice_1844 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_1848: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_563, 3, 0, -1);  view_563 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_150: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_1848, 4);  slice_1848 = None
        permute_493: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 2, 4, 3]);  unsqueeze_150 = None
        permute_495: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_493, [0, 1, 2, 4, 3]);  permute_493 = None
        view_564: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_495, [192, 256, 768]);  permute_495 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_93: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format);  permute_441 = None
        view_508: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_93, [4096, 768]);  clone_93 = None
        permute_444: "f32[768, 768]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_30: "f32[4096, 768]" = torch.ops.aten.mm.default(view_508, permute_444);  view_508 = permute_444 = None
        view_509: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_30, [1024, 4, 768]);  mm_30 = None
        add_107: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_509, arg120_1);  view_509 = arg120_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_559: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_107, [1024, 4, 12, 64]);  add_107 = None
        permute_490: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_559, [1, 0, 2, 3]);  view_559 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_492: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
        view_561: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_492, [48, 1024, 64]);  permute_492 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_30: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_561, [0, 0, 256, 256], -1.0);  view_561 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_47: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_30, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_30 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_151: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_47, 4);  as_strided_47 = None
        permute_494: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_151, [0, 1, 4, 3, 2]);  unsqueeze_151 = None
        permute_496: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_494, [0, 1, 4, 3, 2]);  permute_494 = None
        clone_99: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
        view_565: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_99, [192, 768, 64]);  clone_99 = None
        bmm_15: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_564, view_565);  view_564 = view_565 = None
        view_566: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_15, [48, 4, 256, 1, 64]);  bmm_15 = None
        permute_497: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_566, [0, 1, 2, 4, 3]);  view_566 = None
        view_567: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_497, [48, 4, 256, 64]);  permute_497 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_568: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_567, [4, 12, 1024, 64]);  view_567 = None
        permute_498: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_499: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_498, [1, 0, 2, 3]);  permute_498 = None
        clone_100: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
        view_569: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_100, [1024, 4, 768]);  clone_100 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_500: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_569, [1, 0, 2]);  view_569 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_101: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_570: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_101, [4096, 768]);  clone_101 = None
        permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_31: "f32[4096, 768]" = torch.ops.aten.mm.default(view_570, permute_501);  view_570 = permute_501 = None
        view_571: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_31, [4, 1024, 768]);  mm_31 = None
        add_112: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_571, arg122_1);  view_571 = arg122_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_113: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_112, add_104);  add_112 = add_104 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_28: "f32[4, 1024, 1]" = var_mean_14[0]
        getitem_29: "f32[4, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_62: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_113, getitem_29);  add_113 = getitem_29 = None
        add_114: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_57: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = rsqrt_14 = None
        mul_58: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg123_1);  mul_57 = arg123_1 = None
        add_115: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_58, arg124_1);  mul_58 = arg124_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_572: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_115, [4096, 768])
        permute_502: "f32[768, 3072]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        
        # No stacktrace found for following nodes
        mm_default_9: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_572, permute_502);  view_572 = permute_502 = None
        add_tensor_9: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_9, arg126_1);  mm_default_9 = arg126_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_573: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_9, [4, 1024, 3072]);  add_tensor_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_59: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_573, 0.5)
        mul_60: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_573, 0.7071067811865476);  view_573 = None
        erf_7: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_116: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_61: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_116);  mul_59 = add_116 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_574: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_61, [4096, 3072]);  mul_61 = None
        permute_503: "f32[3072, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        
        # No stacktrace found for following nodes
        mm_default_8: "f32[4096, 768]" = torch.ops.aten.mm.default(view_574, permute_503);  view_574 = permute_503 = None
        add_tensor_8: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_8, arg128_1);  mm_default_8 = arg128_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_575: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [4, 1024, 768]);  add_tensor_8 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_117: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_575, add_115);  view_575 = add_115 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_30: "f32[4, 1024, 1]" = var_mean_15[0]
        getitem_31: "f32[4, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_167: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_168: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        full_default_89: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_72: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1858: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1);  slice_1858 = None
        slice_1854: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1)
        slice_1856: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1854, 3, 256, 9223372036854775807);  slice_1854 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_63: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_117, getitem_31);  add_117 = getitem_31 = None
        add_118: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_62: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = rsqrt_15 = None
        mul_63: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, arg129_1);  mul_62 = arg129_1 = None
        add_119: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_63, arg130_1);  mul_63 = arg130_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_504: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_119, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_104: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format)
        view_576: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_104, [4096, 768]);  clone_104 = None
        permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_32: "f32[4096, 768]" = torch.ops.aten.mm.default(view_576, permute_505);  view_576 = permute_505 = None
        view_577: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_32, [1024, 4, 768]);  mm_32 = None
        add_120: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_577, arg132_1);  view_577 = arg132_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_80: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_120, 8.0);  add_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_588: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_80, [1024, 4, 12, 64]);  div_80 = None
        permute_515: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_588, [1, 0, 2, 3]);  view_588 = None
        permute_516: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_515, [0, 2, 1, 3]);  permute_515 = None
        view_589: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_516, [48, 1024, 64]);  permute_516 = None
        view_590: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_589, [48, 2, 512, 64]);  view_589 = None
        as_strided_50: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_590, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_590 = None
        unsqueeze_154: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_50, 4);  as_strided_50 = None
        permute_517: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_154, [0, 1, 2, 4, 3]);  unsqueeze_154 = None
        permute_518: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_517, [0, 1, 2, 4, 3]);  permute_517 = None
        clone_107: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_591: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_107, [144, 512, 64]);  clone_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_105: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format)
        view_578: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_105, [4096, 768]);  clone_105 = None
        permute_506: "f32[768, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_33: "f32[4096, 768]" = torch.ops.aten.mm.default(view_578, permute_506);  view_578 = permute_506 = None
        view_579: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_33, [1024, 4, 768]);  mm_33 = None
        add_121: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_579, arg134_1);  view_579 = arg134_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_583: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_121, [1024, 4, 12, 64]);  add_121 = None
        permute_509: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_583, [1, 0, 2, 3]);  view_583 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_511: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_509, [0, 2, 1, 3]);  permute_509 = None
        view_585: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_511, [48, 1024, 64]);  permute_511 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_587: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_585, [48, 2, 512, 64]);  view_585 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_49: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_587, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_587 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_153: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_49, 4);  as_strided_49 = None
        permute_513: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_153, [0, 1, 4, 2, 3]);  unsqueeze_153 = None
        permute_519: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_513, [0, 1, 4, 3, 2]);  permute_513 = None
        clone_108: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
        view_592: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_108, [144, 64, 512]);  clone_108 = None
        bmm_16: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_591, view_592);  view_591 = view_592 = None
        view_593: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_16, [48, 3, 512, 1, 512]);  bmm_16 = None
        permute_520: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_593, [0, 1, 2, 4, 3]);  view_593 = None
        view_594: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_520, [48, 3, 512, 512]);  permute_520 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_32: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_594, [0, 0, 0, 1], 0.0);  view_594 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_595: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_32, [48, 3, 512, 513]);  constant_pad_nd_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1851: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_595, 2, 0, 256)
        slice_1852: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1851, 3, 0, 257);  slice_1851 = None
        copy_96: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1856, slice_1852);  slice_1856 = slice_1852 = None
        
        # No stacktrace found for following nodes
        slice_tensor_80: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_72, 1, 0, -1)
        slice_scatter_default_176: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_80, copy_96, 3, 256, 9223372036854775807);  slice_tensor_80 = copy_96 = None
        slice_scatter_default_177: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_72, slice_scatter_default_176, 1, 0, -1);  full_72 = slice_scatter_default_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_163: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_177, 1, -1);  select_163 = None
        select_162: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_177, 1, -1)
        slice_1872: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_162, 2, 256, 9223372036854775807);  select_162 = None
        select_160: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_595, 1, -1)
        slice_1865: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_160, 1, 256, 9223372036854775807);  select_160 = None
        slice_1866: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1865, 2, 0, 257);  slice_1865 = None
        copy_97: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_1872, slice_1866);  slice_1872 = slice_1866 = None
        
        # No stacktrace found for following nodes
        select_int_32: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_177, 1, -1)
        slice_scatter_default_178: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_32, copy_97, 2, 256, 9223372036854775807);  select_int_32 = copy_97 = None
        select_scatter_default_32: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_177, slice_scatter_default_178, 1, -1);  slice_scatter_default_177 = slice_scatter_default_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1891: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_32, 1, 1, 9223372036854775807);  slice_1891 = None
        slice_1887: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_32, 1, 1, 9223372036854775807)
        slice_1889: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1887, 3, 0, 256);  slice_1887 = None
        slice_1880: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_595, 2, -257, -1)
        slice_1881: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1880, 3, 257, 9223372036854775807);  slice_1880 = None
        copy_98: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1889, slice_1881);  slice_1889 = slice_1881 = None
        
        # No stacktrace found for following nodes
        slice_tensor_81: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_32, 1, 1, 9223372036854775807)
        slice_scatter_default_179: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_81, copy_98, 3, 0, 256);  slice_tensor_81 = copy_98 = None
        slice_scatter_default_180: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_32, slice_scatter_default_179, 1, 1, 9223372036854775807);  select_scatter_default_32 = slice_scatter_default_179 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_168: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_180, 1, 0)
        slice_1907: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_168, 1, 1, 256);  select_168 = slice_1907 = None
        select_167: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_180, 1, 0)
        slice_1904: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_167, 1, 1, 256);  select_167 = None
        slice_1905: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1904, 2, 1, 256);  slice_1904 = None
        select_165: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_595, 1, 0);  view_595 = None
        slice_1898: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_165, 1, 0, 255);  select_165 = None
        slice_1899: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1898, 2, -255, 9223372036854775807);  slice_1898 = None
        copy_99: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_1905, slice_1899);  slice_1905 = slice_1899 = None
        
        # No stacktrace found for following nodes
        select_int_33: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_180, 1, 0)
        slice_tensor_82: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_33, 1, 1, 256)
        slice_scatter_default_181: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_82, copy_99, 2, 1, 256);  slice_tensor_82 = copy_99 = None
        slice_scatter_default_182: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_33, slice_scatter_default_181, 1, 1, 256);  select_int_33 = slice_scatter_default_181 = None
        select_scatter_default_33: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_180, slice_scatter_default_182, 1, 0);  slice_scatter_default_180 = slice_scatter_default_182 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_600: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_33, [4, 12, 1024, 513])
        permute_525: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
        slice_1930: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_525, 1, 0, 256);  slice_1930 = None
        view_599: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_33, [4, 12, 1024, 513])
        permute_524: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
        slice_1926: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_524, 1, 0, 256);  permute_524 = None
        slice_1928: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1926, 3, 0, 257);  slice_1926 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_32: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_155: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_32, -2);  iota_32 = None
        iota_33: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_156: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        sub_65: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_155, unsqueeze_156);  unsqueeze_155 = unsqueeze_156 = None
        le_16: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_65, 0);  sub_65 = None
        full_default_80: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_64: "f32[256, 257]" = torch.ops.aten.where.self(le_16, full_default_80, full_default_81);  le_16 = full_default_80 = full_default_81 = None
        rev_32: "f32[256, 257]" = torch.ops.prims.rev.default(where_64, [0]);  where_64 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_157: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_32, 0);  rev_32 = None
        unsqueeze_158: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_32: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_158, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_40: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_32, torch.bool);  expand_32 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_82: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_598: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_33, [4, 12, 1024, 513]);  select_scatter_default_33 = None
        permute_523: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
        slice_1918: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_523, 1, 0, 256);  permute_523 = None
        slice_1920: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1918, 3, 0, 257);  slice_1918 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_65: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_40, full_default_82, slice_1920);  convert_element_type_40 = full_default_82 = slice_1920 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_100: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1928, where_65);  slice_1928 = where_65 = None
        
        # No stacktrace found for following nodes
        slice_tensor_83: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_525, 1, 0, 256)
        slice_scatter_default_183: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_83, copy_100, 3, 0, 257);  slice_tensor_83 = copy_100 = None
        slice_scatter_default_184: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_525, slice_scatter_default_183, 1, 0, 256);  permute_525 = slice_scatter_default_183 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_526: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_184, [0, 2, 1, 3]);  slice_scatter_default_184 = None
        view_601: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_526, [48, 4, 256, 513]);  permute_526 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_605: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_601, [4, 12, 1024, 513])
        permute_530: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_605, [0, 2, 1, 3]);  view_605 = None
        slice_1953: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_530, 1, -256, 9223372036854775807);  slice_1953 = None
        view_604: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_601, [4, 12, 1024, 513])
        permute_529: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
        slice_1949: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_529, 1, -256, 9223372036854775807);  permute_529 = None
        slice_1951: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1949, 3, -257, 9223372036854775807);  slice_1949 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_33: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_158, [1, 3]);  unsqueeze_158 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_33: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_33, [4, 256, 12, 257]);  rev_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_41: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_33, torch.bool);  expand_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_83: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_603: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_601, [4, 12, 1024, 513]);  view_601 = None
        permute_528: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
        slice_1941: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_528, 1, -256, 9223372036854775807);  permute_528 = None
        slice_1943: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1941, 3, -257, 9223372036854775807);  slice_1941 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_66: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_41, full_default_83, slice_1943);  convert_element_type_41 = full_default_83 = slice_1943 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_101: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1951, where_66);  slice_1951 = where_66 = None
        
        # No stacktrace found for following nodes
        slice_tensor_84: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_530, 1, -256, 9223372036854775807)
        slice_scatter_default_185: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_84, copy_101, 3, -257, 9223372036854775807);  slice_tensor_84 = copy_101 = None
        slice_scatter_default_186: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_530, slice_scatter_default_185, 1, -256, 9223372036854775807);  permute_530 = slice_scatter_default_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_531: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_186, [0, 2, 1, 3]);  slice_scatter_default_186 = None
        view_606: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_531, [48, 4, 256, 513]);  permute_531 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_626: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_606, [4, 12, 1024, 513]);  view_606 = None
        permute_549: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_77: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1970: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1);  slice_1970 = None
        slice_1966: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1)
        slice_1968: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1966, 3, 256, 9223372036854775807);  slice_1966 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_76: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_533: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_76, [0, 2, 1, 3]);  full_76 = None
        view_608: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_533, [4, 1024, 1]);  permute_533 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_610: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_608, [4, 2, 512, 1]);  view_608 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_51: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_610, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_610 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_161: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_51, 4);  as_strided_51 = None
        permute_535: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_161, [0, 1, 2, 4, 3]);  unsqueeze_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_84: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_8: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_159: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_8, 2);  ne_8 = None
        unsqueeze_160: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_42: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_160, torch.float32)
        where_67: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_160, full_default_84, convert_element_type_42);  unsqueeze_160 = full_default_84 = convert_element_type_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_534: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_67, [0, 2, 1, 3]);  where_67 = None
        view_609: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_534, [4, 1024, 1]);  permute_534 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_611: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_609, [4, 2, 512, 1]);  view_609 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_52: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_611, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_611 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_162: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_52, 4);  as_strided_52 = None
        permute_536: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_162, [0, 1, 4, 2, 3]);  unsqueeze_162 = None
        mul_64: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_535, permute_536);  permute_535 = permute_536 = None
        view_612: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_64, [4, 3, 512, 512]);  mul_64 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_33: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_612, [0, 0, 0, 1], 0.0);  view_612 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_613: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_33, [4, 3, 512, 513]);  constant_pad_nd_33 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1963: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_613, 2, 0, 256)
        slice_1964: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1963, 3, 0, 257);  slice_1963 = None
        copy_102: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_1968, slice_1964);  slice_1968 = slice_1964 = None
        
        # No stacktrace found for following nodes
        slice_tensor_85: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_77, 1, 0, -1)
        slice_scatter_default_187: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_85, copy_102, 3, 256, 9223372036854775807);  slice_tensor_85 = copy_102 = None
        slice_scatter_default_188: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_77, slice_scatter_default_187, 1, 0, -1);  full_77 = slice_scatter_default_187 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_173: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_188, 1, -1);  select_173 = None
        select_172: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_188, 1, -1)
        slice_1984: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_172, 2, 256, 9223372036854775807);  select_172 = None
        select_170: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_613, 1, -1)
        slice_1977: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_170, 1, 256, 9223372036854775807);  select_170 = None
        slice_1978: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1977, 2, 0, 257);  slice_1977 = None
        copy_103: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_1984, slice_1978);  slice_1984 = slice_1978 = None
        
        # No stacktrace found for following nodes
        select_int_34: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_188, 1, -1)
        slice_scatter_default_189: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_34, copy_103, 2, 256, 9223372036854775807);  select_int_34 = copy_103 = None
        select_scatter_default_34: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_188, slice_scatter_default_189, 1, -1);  slice_scatter_default_188 = slice_scatter_default_189 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2003: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_34, 1, 1, 9223372036854775807);  slice_2003 = None
        slice_1999: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_34, 1, 1, 9223372036854775807)
        slice_2001: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1999, 3, 0, 256);  slice_1999 = None
        slice_1992: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_613, 2, -257, -1)
        slice_1993: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1992, 3, 257, 9223372036854775807);  slice_1992 = None
        copy_104: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2001, slice_1993);  slice_2001 = slice_1993 = None
        
        # No stacktrace found for following nodes
        slice_tensor_86: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_34, 1, 1, 9223372036854775807)
        slice_scatter_default_190: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_86, copy_104, 3, 0, 256);  slice_tensor_86 = copy_104 = None
        slice_scatter_default_191: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_34, slice_scatter_default_190, 1, 1, 9223372036854775807);  select_scatter_default_34 = slice_scatter_default_190 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_178: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_191, 1, 0)
        slice_2019: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_178, 1, 1, 256);  select_178 = slice_2019 = None
        select_177: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_191, 1, 0)
        slice_2016: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_177, 1, 1, 256);  select_177 = None
        slice_2017: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2016, 2, 1, 256);  slice_2016 = None
        select_175: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_613, 1, 0);  view_613 = None
        slice_2010: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_175, 1, 0, 255);  select_175 = None
        slice_2011: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2010, 2, -255, 9223372036854775807);  slice_2010 = None
        copy_105: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_2017, slice_2011);  slice_2017 = slice_2011 = None
        
        # No stacktrace found for following nodes
        select_int_35: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_191, 1, 0)
        slice_tensor_87: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_35, 1, 1, 256)
        slice_scatter_default_192: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_87, copy_105, 2, 1, 256);  slice_tensor_87 = copy_105 = None
        slice_scatter_default_193: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_35, slice_scatter_default_192, 1, 1, 256);  select_int_35 = slice_scatter_default_192 = None
        select_scatter_default_35: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_191, slice_scatter_default_193, 1, 0);  slice_scatter_default_191 = slice_scatter_default_193 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_618: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_35, [4, 1, 1024, 513])
        permute_541: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_618, [0, 2, 1, 3]);  view_618 = None
        slice_2042: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_541, 1, 0, 256);  slice_2042 = None
        view_617: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_35, [4, 1, 1024, 513])
        permute_540: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_617, [0, 2, 1, 3]);  view_617 = None
        slice_2038: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_540, 1, 0, 256);  permute_540 = None
        slice_2040: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2038, 3, 0, 257);  slice_2038 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_34: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_163: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_34, -2);  iota_34 = None
        iota_35: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_164: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
        sub_67: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_163, unsqueeze_164);  unsqueeze_163 = unsqueeze_164 = None
        le_17: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_67, 0);  sub_67 = None
        full_default_85: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_68: "f32[256, 257]" = torch.ops.aten.where.self(le_17, full_default_85, full_default_86);  le_17 = full_default_85 = full_default_86 = None
        rev_34: "f32[256, 257]" = torch.ops.prims.rev.default(where_68, [0]);  where_68 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_165: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_34, 0);  rev_34 = None
        unsqueeze_166: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_34: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_166, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_43: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_34, torch.bool);  expand_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_87: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_616: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_35, [4, 1, 1024, 513]);  select_scatter_default_35 = None
        permute_539: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
        slice_2030: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_539, 1, 0, 256);  permute_539 = None
        slice_2032: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2030, 3, 0, 257);  slice_2030 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_69: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_43, full_default_87, slice_2032);  convert_element_type_43 = full_default_87 = slice_2032 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_106: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2040, where_69);  slice_2040 = where_69 = None
        
        # No stacktrace found for following nodes
        slice_tensor_88: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_541, 1, 0, 256)
        slice_scatter_default_194: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_88, copy_106, 3, 0, 257);  slice_tensor_88 = copy_106 = None
        slice_scatter_default_195: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_541, slice_scatter_default_194, 1, 0, 256);  permute_541 = slice_scatter_default_194 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_542: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_195, [0, 2, 1, 3]);  slice_scatter_default_195 = None
        view_619: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_542, [4, 4, 256, 513]);  permute_542 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_623: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_619, [4, 1, 1024, 513])
        permute_546: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None
        slice_2065: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_546, 1, -256, 9223372036854775807);  slice_2065 = None
        view_622: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_619, [4, 1, 1024, 513])
        permute_545: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_622, [0, 2, 1, 3]);  view_622 = None
        slice_2061: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_545, 1, -256, 9223372036854775807);  permute_545 = None
        slice_2063: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2061, 3, -257, 9223372036854775807);  slice_2061 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_35: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_166, [1, 3]);  unsqueeze_166 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_35: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_35, [4, 256, 1, 257]);  rev_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_44: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_35, torch.bool);  expand_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_88: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_621: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_619, [4, 1, 1024, 513]);  view_619 = None
        permute_544: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_621, [0, 2, 1, 3]);  view_621 = None
        slice_2053: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_544, 1, -256, 9223372036854775807);  permute_544 = None
        slice_2055: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2053, 3, -257, 9223372036854775807);  slice_2053 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_70: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_44, full_default_88, slice_2055);  convert_element_type_44 = full_default_88 = slice_2055 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_107: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2063, where_70);  slice_2063 = where_70 = None
        
        # No stacktrace found for following nodes
        slice_tensor_89: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_546, 1, -256, 9223372036854775807)
        slice_scatter_default_196: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_89, copy_107, 3, -257, 9223372036854775807);  slice_tensor_89 = copy_107 = None
        slice_scatter_default_197: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_546, slice_scatter_default_196, 1, -256, 9223372036854775807);  permute_546 = slice_scatter_default_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_547: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_197, [0, 2, 1, 3]);  slice_scatter_default_197 = None
        view_624: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_547, [4, 4, 256, 513]);  permute_547 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_627: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_624, [4, 1, 1024, 513]);  view_624 = None
        permute_550: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
        add_125: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_549, permute_550);  permute_549 = permute_550 = None
        permute_551: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_125, [0, 2, 1, 3]);  add_125 = None
        view_629: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_551, [48, 4, 256, 513]);  permute_551 = None
        view_630: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_629, [4, 12, 1024, 513]);  view_629 = None
        permute_552: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_109: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
        amax_8: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_109, [-1], True)
        sub_68: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_109, amax_8);  clone_109 = amax_8 = None
        exp_8: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
        sum_9: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_87: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_71: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_168, full_default_89, div_87);  unsqueeze_168 = full_default_89 = div_87 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_554: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_71, [0, 2, 1, 3]);  where_71 = None
        clone_111: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_554, memory_format = torch.contiguous_format);  permute_554 = None
        view_632: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_111, [48, 4, 256, 513]);  clone_111 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_35: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_632, [0, 257], 0.0);  view_632 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_634: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_35, [48, 4, -1]);  constant_pad_nd_35 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_2075: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_634, 2, 0, -256);  view_634 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_635: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_2075, [48, 4, 256, 769]);  slice_2075 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_2079: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_635, 3, 0, -1);  view_635 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_169: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2079, 4);  slice_2079 = None
        permute_556: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_169, [0, 1, 2, 4, 3]);  unsqueeze_169 = None
        permute_558: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_556, [0, 1, 2, 4, 3]);  permute_556 = None
        view_636: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_558, [192, 256, 768]);  permute_558 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_106: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format);  permute_504 = None
        view_580: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_106, [4096, 768]);  clone_106 = None
        permute_507: "f32[768, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_34: "f32[4096, 768]" = torch.ops.aten.mm.default(view_580, permute_507);  view_580 = permute_507 = None
        view_581: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_34, [1024, 4, 768]);  mm_34 = None
        add_122: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_581, arg136_1);  view_581 = arg136_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_631: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_122, [1024, 4, 12, 64]);  add_122 = None
        permute_553: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_631, [1, 0, 2, 3]);  view_631 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_555: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_553, [0, 2, 1, 3]);  permute_553 = None
        view_633: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_555, [48, 1024, 64]);  permute_555 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_34: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_633, [0, 0, 256, 256], -1.0);  view_633 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_53: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_34, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_34 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_170: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_53, 4);  as_strided_53 = None
        permute_557: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_170, [0, 1, 4, 3, 2]);  unsqueeze_170 = None
        permute_559: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_557, [0, 1, 4, 3, 2]);  permute_557 = None
        clone_112: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
        view_637: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_112, [192, 768, 64]);  clone_112 = None
        bmm_17: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_636, view_637);  view_636 = view_637 = None
        view_638: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_17, [48, 4, 256, 1, 64]);  bmm_17 = None
        permute_560: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_638, [0, 1, 2, 4, 3]);  view_638 = None
        view_639: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_560, [48, 4, 256, 64]);  permute_560 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_640: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_639, [4, 12, 1024, 64]);  view_639 = None
        permute_561: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_562: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_561, [1, 0, 2, 3]);  permute_561 = None
        clone_113: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_562, memory_format = torch.contiguous_format);  permute_562 = None
        view_641: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_113, [1024, 4, 768]);  clone_113 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_563: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_641, [1, 0, 2]);  view_641 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_114: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_563, memory_format = torch.contiguous_format);  permute_563 = None
        view_642: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_114, [4096, 768]);  clone_114 = None
        permute_564: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_35: "f32[4096, 768]" = torch.ops.aten.mm.default(view_642, permute_564);  view_642 = permute_564 = None
        view_643: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_35, [4, 1024, 768]);  mm_35 = None
        add_127: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_643, arg138_1);  view_643 = arg138_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_128: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_127, add_119);  add_127 = add_119 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_32: "f32[4, 1024, 1]" = var_mean_16[0]
        getitem_33: "f32[4, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_70: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_128, getitem_33);  add_128 = getitem_33 = None
        add_129: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_65: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_16);  sub_70 = rsqrt_16 = None
        mul_66: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg139_1);  mul_65 = arg139_1 = None
        add_130: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_66, arg140_1);  mul_66 = arg140_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_644: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_130, [4096, 768])
        permute_565: "f32[768, 3072]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        
        # No stacktrace found for following nodes
        mm_default_7: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_644, permute_565);  view_644 = permute_565 = None
        add_tensor_7: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_7, arg142_1);  mm_default_7 = arg142_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_645: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_7, [4, 1024, 3072]);  add_tensor_7 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_67: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_645, 0.5)
        mul_68: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_645, 0.7071067811865476);  view_645 = None
        erf_8: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_131: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_69: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_131);  mul_67 = add_131 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_646: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_69, [4096, 3072]);  mul_69 = None
        permute_566: "f32[3072, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        
        # No stacktrace found for following nodes
        mm_default_6: "f32[4096, 768]" = torch.ops.aten.mm.default(view_646, permute_566);  view_646 = permute_566 = None
        add_tensor_6: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg144_1);  mm_default_6 = arg144_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_647: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [4, 1024, 768]);  add_tensor_6 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_132: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_647, add_130);  view_647 = add_130 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
        getitem_34: "f32[4, 1024, 1]" = var_mean_17[0]
        getitem_35: "f32[4, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_186: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_187: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
        full_default_99: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_81: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2089: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1);  slice_2089 = None
        slice_2085: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1)
        slice_2087: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2085, 3, 256, 9223372036854775807);  slice_2085 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_71: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_132, getitem_35);  add_132 = getitem_35 = None
        add_133: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_70: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_17);  sub_71 = rsqrt_17 = None
        mul_71: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg145_1);  mul_70 = arg145_1 = None
        add_134: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_71, arg146_1);  mul_71 = arg146_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_567: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_134, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_117: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_567, memory_format = torch.contiguous_format)
        view_648: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_117, [4096, 768]);  clone_117 = None
        permute_568: "f32[768, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_36: "f32[4096, 768]" = torch.ops.aten.mm.default(view_648, permute_568);  view_648 = permute_568 = None
        view_649: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_36, [1024, 4, 768]);  mm_36 = None
        add_135: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_649, arg148_1);  view_649 = arg148_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_90: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_135, 8.0);  add_135 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_660: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_90, [1024, 4, 12, 64]);  div_90 = None
        permute_578: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_660, [1, 0, 2, 3]);  view_660 = None
        permute_579: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_578, [0, 2, 1, 3]);  permute_578 = None
        view_661: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_579, [48, 1024, 64]);  permute_579 = None
        view_662: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_661, [48, 2, 512, 64]);  view_661 = None
        as_strided_56: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_662, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_662 = None
        unsqueeze_173: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_56, 4);  as_strided_56 = None
        permute_580: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_173, [0, 1, 2, 4, 3]);  unsqueeze_173 = None
        permute_581: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_580, [0, 1, 2, 4, 3]);  permute_580 = None
        clone_120: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
        view_663: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_120, [144, 512, 64]);  clone_120 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_118: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_567, memory_format = torch.contiguous_format)
        view_650: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_118, [4096, 768]);  clone_118 = None
        permute_569: "f32[768, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_37: "f32[4096, 768]" = torch.ops.aten.mm.default(view_650, permute_569);  view_650 = permute_569 = None
        view_651: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_37, [1024, 4, 768]);  mm_37 = None
        add_136: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_651, arg150_1);  view_651 = arg150_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_655: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_136, [1024, 4, 12, 64]);  add_136 = None
        permute_572: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_655, [1, 0, 2, 3]);  view_655 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_574: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_572, [0, 2, 1, 3]);  permute_572 = None
        view_657: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_574, [48, 1024, 64]);  permute_574 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_659: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_657, [48, 2, 512, 64]);  view_657 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_55: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_659, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_172: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_55, 4);  as_strided_55 = None
        permute_576: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_172, [0, 1, 4, 2, 3]);  unsqueeze_172 = None
        permute_582: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_576, [0, 1, 4, 3, 2]);  permute_576 = None
        clone_121: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_582, memory_format = torch.contiguous_format);  permute_582 = None
        view_664: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_121, [144, 64, 512]);  clone_121 = None
        bmm_18: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_663, view_664);  view_663 = view_664 = None
        view_665: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_18, [48, 3, 512, 1, 512]);  bmm_18 = None
        permute_583: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_665, [0, 1, 2, 4, 3]);  view_665 = None
        view_666: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_583, [48, 3, 512, 512]);  permute_583 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_36: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_666, [0, 0, 0, 1], 0.0);  view_666 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_667: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_36, [48, 3, 512, 513]);  constant_pad_nd_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2082: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_667, 2, 0, 256)
        slice_2083: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2082, 3, 0, 257);  slice_2082 = None
        copy_108: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2087, slice_2083);  slice_2087 = slice_2083 = None
        
        # No stacktrace found for following nodes
        slice_tensor_90: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_81, 1, 0, -1)
        slice_scatter_default_198: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_90, copy_108, 3, 256, 9223372036854775807);  slice_tensor_90 = copy_108 = None
        slice_scatter_default_199: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_81, slice_scatter_default_198, 1, 0, -1);  full_81 = slice_scatter_default_198 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_183: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_199, 1, -1);  select_183 = None
        select_182: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_199, 1, -1)
        slice_2103: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_182, 2, 256, 9223372036854775807);  select_182 = None
        select_180: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_667, 1, -1)
        slice_2096: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_180, 1, 256, 9223372036854775807);  select_180 = None
        slice_2097: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2096, 2, 0, 257);  slice_2096 = None
        copy_109: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_2103, slice_2097);  slice_2103 = slice_2097 = None
        
        # No stacktrace found for following nodes
        select_int_36: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_199, 1, -1)
        slice_scatter_default_200: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_36, copy_109, 2, 256, 9223372036854775807);  select_int_36 = copy_109 = None
        select_scatter_default_36: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_199, slice_scatter_default_200, 1, -1);  slice_scatter_default_199 = slice_scatter_default_200 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2122: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_36, 1, 1, 9223372036854775807);  slice_2122 = None
        slice_2118: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_36, 1, 1, 9223372036854775807)
        slice_2120: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2118, 3, 0, 256);  slice_2118 = None
        slice_2111: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_667, 2, -257, -1)
        slice_2112: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2111, 3, 257, 9223372036854775807);  slice_2111 = None
        copy_110: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2120, slice_2112);  slice_2120 = slice_2112 = None
        
        # No stacktrace found for following nodes
        slice_tensor_91: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_36, 1, 1, 9223372036854775807)
        slice_scatter_default_201: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_91, copy_110, 3, 0, 256);  slice_tensor_91 = copy_110 = None
        slice_scatter_default_202: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_36, slice_scatter_default_201, 1, 1, 9223372036854775807);  select_scatter_default_36 = slice_scatter_default_201 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_188: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_202, 1, 0)
        slice_2138: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_188, 1, 1, 256);  select_188 = slice_2138 = None
        select_187: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_202, 1, 0)
        slice_2135: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_187, 1, 1, 256);  select_187 = None
        slice_2136: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2135, 2, 1, 256);  slice_2135 = None
        select_185: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_667, 1, 0);  view_667 = None
        slice_2129: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_185, 1, 0, 255);  select_185 = None
        slice_2130: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2129, 2, -255, 9223372036854775807);  slice_2129 = None
        copy_111: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_2136, slice_2130);  slice_2136 = slice_2130 = None
        
        # No stacktrace found for following nodes
        select_int_37: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_202, 1, 0)
        slice_tensor_92: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_37, 1, 1, 256)
        slice_scatter_default_203: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_92, copy_111, 2, 1, 256);  slice_tensor_92 = copy_111 = None
        slice_scatter_default_204: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_37, slice_scatter_default_203, 1, 1, 256);  select_int_37 = slice_scatter_default_203 = None
        select_scatter_default_37: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_202, slice_scatter_default_204, 1, 0);  slice_scatter_default_202 = slice_scatter_default_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_672: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_37, [4, 12, 1024, 513])
        permute_588: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
        slice_2161: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_588, 1, 0, 256);  slice_2161 = None
        view_671: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_37, [4, 12, 1024, 513])
        permute_587: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None
        slice_2157: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_587, 1, 0, 256);  permute_587 = None
        slice_2159: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2157, 3, 0, 257);  slice_2157 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_36: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_174: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_36, -2);  iota_36 = None
        iota_37: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_175: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        sub_73: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_174, unsqueeze_175);  unsqueeze_174 = unsqueeze_175 = None
        le_18: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_73, 0);  sub_73 = None
        full_default_90: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_91: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_72: "f32[256, 257]" = torch.ops.aten.where.self(le_18, full_default_90, full_default_91);  le_18 = full_default_90 = full_default_91 = None
        rev_36: "f32[256, 257]" = torch.ops.prims.rev.default(where_72, [0]);  where_72 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_176: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_36, 0);  rev_36 = None
        unsqueeze_177: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_36: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_177, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_45: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_36, torch.bool);  expand_36 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_92: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_670: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_37, [4, 12, 1024, 513]);  select_scatter_default_37 = None
        permute_586: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_670, [0, 2, 1, 3]);  view_670 = None
        slice_2149: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_586, 1, 0, 256);  permute_586 = None
        slice_2151: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2149, 3, 0, 257);  slice_2149 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_73: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_45, full_default_92, slice_2151);  convert_element_type_45 = full_default_92 = slice_2151 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_112: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2159, where_73);  slice_2159 = where_73 = None
        
        # No stacktrace found for following nodes
        slice_tensor_93: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_588, 1, 0, 256)
        slice_scatter_default_205: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_93, copy_112, 3, 0, 257);  slice_tensor_93 = copy_112 = None
        slice_scatter_default_206: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_588, slice_scatter_default_205, 1, 0, 256);  permute_588 = slice_scatter_default_205 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_589: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_206, [0, 2, 1, 3]);  slice_scatter_default_206 = None
        view_673: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_589, [48, 4, 256, 513]);  permute_589 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_677: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_673, [4, 12, 1024, 513])
        permute_593: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_677, [0, 2, 1, 3]);  view_677 = None
        slice_2184: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_593, 1, -256, 9223372036854775807);  slice_2184 = None
        view_676: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_673, [4, 12, 1024, 513])
        permute_592: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None
        slice_2180: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_592, 1, -256, 9223372036854775807);  permute_592 = None
        slice_2182: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2180, 3, -257, 9223372036854775807);  slice_2180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_37: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_177, [1, 3]);  unsqueeze_177 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_37: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_37, [4, 256, 12, 257]);  rev_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_46: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_37, torch.bool);  expand_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_93: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_675: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_673, [4, 12, 1024, 513]);  view_673 = None
        permute_591: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_675, [0, 2, 1, 3]);  view_675 = None
        slice_2172: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_591, 1, -256, 9223372036854775807);  permute_591 = None
        slice_2174: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2172, 3, -257, 9223372036854775807);  slice_2172 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_74: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_46, full_default_93, slice_2174);  convert_element_type_46 = full_default_93 = slice_2174 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_113: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2182, where_74);  slice_2182 = where_74 = None
        
        # No stacktrace found for following nodes
        slice_tensor_94: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_593, 1, -256, 9223372036854775807)
        slice_scatter_default_207: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_94, copy_113, 3, -257, 9223372036854775807);  slice_tensor_94 = copy_113 = None
        slice_scatter_default_208: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_593, slice_scatter_default_207, 1, -256, 9223372036854775807);  permute_593 = slice_scatter_default_207 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_594: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_208, [0, 2, 1, 3]);  slice_scatter_default_208 = None
        view_678: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_594, [48, 4, 256, 513]);  permute_594 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_698: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_678, [4, 12, 1024, 513]);  view_678 = None
        permute_612: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_86: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2201: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1);  slice_2201 = None
        slice_2197: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1)
        slice_2199: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2197, 3, 256, 9223372036854775807);  slice_2197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_85: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_596: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_85, [0, 2, 1, 3]);  full_85 = None
        view_680: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_596, [4, 1024, 1]);  permute_596 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_682: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_680, [4, 2, 512, 1]);  view_680 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_57: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_682, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_682 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_180: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_57, 4);  as_strided_57 = None
        permute_598: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_180, [0, 1, 2, 4, 3]);  unsqueeze_180 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_94: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_9: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_178: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_9, 2);  ne_9 = None
        unsqueeze_179: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_47: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_179, torch.float32)
        where_75: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_179, full_default_94, convert_element_type_47);  unsqueeze_179 = full_default_94 = convert_element_type_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_597: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_75, [0, 2, 1, 3]);  where_75 = None
        view_681: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_597, [4, 1024, 1]);  permute_597 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_683: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_681, [4, 2, 512, 1]);  view_681 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_58: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_683, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_683 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_181: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_58, 4);  as_strided_58 = None
        permute_599: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_181, [0, 1, 4, 2, 3]);  unsqueeze_181 = None
        mul_72: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_598, permute_599);  permute_598 = permute_599 = None
        view_684: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_72, [4, 3, 512, 512]);  mul_72 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_37: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_684, [0, 0, 0, 1], 0.0);  view_684 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_685: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_37, [4, 3, 512, 513]);  constant_pad_nd_37 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2194: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_685, 2, 0, 256)
        slice_2195: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2194, 3, 0, 257);  slice_2194 = None
        copy_114: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2199, slice_2195);  slice_2199 = slice_2195 = None
        
        # No stacktrace found for following nodes
        slice_tensor_95: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_86, 1, 0, -1)
        slice_scatter_default_209: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_95, copy_114, 3, 256, 9223372036854775807);  slice_tensor_95 = copy_114 = None
        slice_scatter_default_210: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_86, slice_scatter_default_209, 1, 0, -1);  full_86 = slice_scatter_default_209 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_193: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_210, 1, -1);  select_193 = None
        select_192: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_210, 1, -1)
        slice_2215: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_192, 2, 256, 9223372036854775807);  select_192 = None
        select_190: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_685, 1, -1)
        slice_2208: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_190, 1, 256, 9223372036854775807);  select_190 = None
        slice_2209: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2208, 2, 0, 257);  slice_2208 = None
        copy_115: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_2215, slice_2209);  slice_2215 = slice_2209 = None
        
        # No stacktrace found for following nodes
        select_int_38: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_210, 1, -1)
        slice_scatter_default_211: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_38, copy_115, 2, 256, 9223372036854775807);  select_int_38 = copy_115 = None
        select_scatter_default_38: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_210, slice_scatter_default_211, 1, -1);  slice_scatter_default_210 = slice_scatter_default_211 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2234: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_38, 1, 1, 9223372036854775807);  slice_2234 = None
        slice_2230: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_38, 1, 1, 9223372036854775807)
        slice_2232: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2230, 3, 0, 256);  slice_2230 = None
        slice_2223: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_685, 2, -257, -1)
        slice_2224: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2223, 3, 257, 9223372036854775807);  slice_2223 = None
        copy_116: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2232, slice_2224);  slice_2232 = slice_2224 = None
        
        # No stacktrace found for following nodes
        slice_tensor_96: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_38, 1, 1, 9223372036854775807)
        slice_scatter_default_212: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_96, copy_116, 3, 0, 256);  slice_tensor_96 = copy_116 = None
        slice_scatter_default_213: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_38, slice_scatter_default_212, 1, 1, 9223372036854775807);  select_scatter_default_38 = slice_scatter_default_212 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_198: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_213, 1, 0)
        slice_2250: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_198, 1, 1, 256);  select_198 = slice_2250 = None
        select_197: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_213, 1, 0)
        slice_2247: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_197, 1, 1, 256);  select_197 = None
        slice_2248: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2247, 2, 1, 256);  slice_2247 = None
        select_195: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_685, 1, 0);  view_685 = None
        slice_2241: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_195, 1, 0, 255);  select_195 = None
        slice_2242: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2241, 2, -255, 9223372036854775807);  slice_2241 = None
        copy_117: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_2248, slice_2242);  slice_2248 = slice_2242 = None
        
        # No stacktrace found for following nodes
        select_int_39: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_213, 1, 0)
        slice_tensor_97: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_39, 1, 1, 256)
        slice_scatter_default_214: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_97, copy_117, 2, 1, 256);  slice_tensor_97 = copy_117 = None
        slice_scatter_default_215: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_39, slice_scatter_default_214, 1, 1, 256);  select_int_39 = slice_scatter_default_214 = None
        select_scatter_default_39: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_213, slice_scatter_default_215, 1, 0);  slice_scatter_default_213 = slice_scatter_default_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_690: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_39, [4, 1, 1024, 513])
        permute_604: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None
        slice_2273: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_604, 1, 0, 256);  slice_2273 = None
        view_689: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_39, [4, 1, 1024, 513])
        permute_603: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_689, [0, 2, 1, 3]);  view_689 = None
        slice_2269: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_603, 1, 0, 256);  permute_603 = None
        slice_2271: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2269, 3, 0, 257);  slice_2269 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_38: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_182: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_38, -2);  iota_38 = None
        iota_39: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_183: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
        sub_75: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_182, unsqueeze_183);  unsqueeze_182 = unsqueeze_183 = None
        le_19: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_75, 0);  sub_75 = None
        full_default_95: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_76: "f32[256, 257]" = torch.ops.aten.where.self(le_19, full_default_95, full_default_96);  le_19 = full_default_95 = full_default_96 = None
        rev_38: "f32[256, 257]" = torch.ops.prims.rev.default(where_76, [0]);  where_76 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_184: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_38, 0);  rev_38 = None
        unsqueeze_185: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_38: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_185, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_48: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_38, torch.bool);  expand_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_97: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_688: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_39, [4, 1, 1024, 513]);  select_scatter_default_39 = None
        permute_602: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None
        slice_2261: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_602, 1, 0, 256);  permute_602 = None
        slice_2263: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2261, 3, 0, 257);  slice_2261 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_77: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_48, full_default_97, slice_2263);  convert_element_type_48 = full_default_97 = slice_2263 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_118: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2271, where_77);  slice_2271 = where_77 = None
        
        # No stacktrace found for following nodes
        slice_tensor_98: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_604, 1, 0, 256)
        slice_scatter_default_216: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_98, copy_118, 3, 0, 257);  slice_tensor_98 = copy_118 = None
        slice_scatter_default_217: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_604, slice_scatter_default_216, 1, 0, 256);  permute_604 = slice_scatter_default_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_605: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_217, [0, 2, 1, 3]);  slice_scatter_default_217 = None
        view_691: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_605, [4, 4, 256, 513]);  permute_605 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_695: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_691, [4, 1, 1024, 513])
        permute_609: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None
        slice_2296: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_609, 1, -256, 9223372036854775807);  slice_2296 = None
        view_694: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_691, [4, 1, 1024, 513])
        permute_608: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
        slice_2292: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_608, 1, -256, 9223372036854775807);  permute_608 = None
        slice_2294: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2292, 3, -257, 9223372036854775807);  slice_2292 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_39: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_185, [1, 3]);  unsqueeze_185 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_39: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_39, [4, 256, 1, 257]);  rev_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_49: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_39, torch.bool);  expand_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_98: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_693: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_691, [4, 1, 1024, 513]);  view_691 = None
        permute_607: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
        slice_2284: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_607, 1, -256, 9223372036854775807);  permute_607 = None
        slice_2286: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2284, 3, -257, 9223372036854775807);  slice_2284 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_78: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_49, full_default_98, slice_2286);  convert_element_type_49 = full_default_98 = slice_2286 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_119: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2294, where_78);  slice_2294 = where_78 = None
        
        # No stacktrace found for following nodes
        slice_tensor_99: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_609, 1, -256, 9223372036854775807)
        slice_scatter_default_218: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_99, copy_119, 3, -257, 9223372036854775807);  slice_tensor_99 = copy_119 = None
        slice_scatter_default_219: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_609, slice_scatter_default_218, 1, -256, 9223372036854775807);  permute_609 = slice_scatter_default_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_610: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_219, [0, 2, 1, 3]);  slice_scatter_default_219 = None
        view_696: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_610, [4, 4, 256, 513]);  permute_610 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_699: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_696, [4, 1, 1024, 513]);  view_696 = None
        permute_613: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None
        add_140: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_612, permute_613);  permute_612 = permute_613 = None
        permute_614: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_140, [0, 2, 1, 3]);  add_140 = None
        view_701: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_614, [48, 4, 256, 513]);  permute_614 = None
        view_702: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_701, [4, 12, 1024, 513]);  view_701 = None
        permute_615: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_702, [0, 2, 1, 3]);  view_702 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_122: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
        amax_9: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_122, [-1], True)
        sub_76: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_122, amax_9);  clone_122 = amax_9 = None
        exp_9: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_10: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_97: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_79: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_187, full_default_99, div_97);  unsqueeze_187 = full_default_99 = div_97 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_617: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_79, [0, 2, 1, 3]);  where_79 = None
        clone_124: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
        view_704: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_124, [48, 4, 256, 513]);  clone_124 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_39: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_704, [0, 257], 0.0);  view_704 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_706: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_39, [48, 4, -1]);  constant_pad_nd_39 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_2306: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_706, 2, 0, -256);  view_706 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_707: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_2306, [48, 4, 256, 769]);  slice_2306 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_2310: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_707, 3, 0, -1);  view_707 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_188: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2310, 4);  slice_2310 = None
        permute_619: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_188, [0, 1, 2, 4, 3]);  unsqueeze_188 = None
        permute_621: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_619, [0, 1, 2, 4, 3]);  permute_619 = None
        view_708: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_621, [192, 256, 768]);  permute_621 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_119: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_567, memory_format = torch.contiguous_format);  permute_567 = None
        view_652: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_119, [4096, 768]);  clone_119 = None
        permute_570: "f32[768, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_38: "f32[4096, 768]" = torch.ops.aten.mm.default(view_652, permute_570);  view_652 = permute_570 = None
        view_653: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_38, [1024, 4, 768]);  mm_38 = None
        add_137: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_653, arg152_1);  view_653 = arg152_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_703: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_137, [1024, 4, 12, 64]);  add_137 = None
        permute_616: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_703, [1, 0, 2, 3]);  view_703 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_618: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_616, [0, 2, 1, 3]);  permute_616 = None
        view_705: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_618, [48, 1024, 64]);  permute_618 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_38: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_705, [0, 0, 256, 256], -1.0);  view_705 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_59: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_38, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_38 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_189: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_59, 4);  as_strided_59 = None
        permute_620: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_189, [0, 1, 4, 3, 2]);  unsqueeze_189 = None
        permute_622: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_620, [0, 1, 4, 3, 2]);  permute_620 = None
        clone_125: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
        view_709: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_125, [192, 768, 64]);  clone_125 = None
        bmm_19: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_708, view_709);  view_708 = view_709 = None
        view_710: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_19, [48, 4, 256, 1, 64]);  bmm_19 = None
        permute_623: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_710, [0, 1, 2, 4, 3]);  view_710 = None
        view_711: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_623, [48, 4, 256, 64]);  permute_623 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_712: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_711, [4, 12, 1024, 64]);  view_711 = None
        permute_624: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_712, [0, 2, 1, 3]);  view_712 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_625: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_624, [1, 0, 2, 3]);  permute_624 = None
        clone_126: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_625, memory_format = torch.contiguous_format);  permute_625 = None
        view_713: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_126, [1024, 4, 768]);  clone_126 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_626: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_713, [1, 0, 2]);  view_713 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_127: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
        view_714: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_127, [4096, 768]);  clone_127 = None
        permute_627: "f32[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        mm_39: "f32[4096, 768]" = torch.ops.aten.mm.default(view_714, permute_627);  view_714 = permute_627 = None
        view_715: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_39, [4, 1024, 768]);  mm_39 = None
        add_142: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_715, arg154_1);  view_715 = arg154_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_143: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_142, add_134);  add_142 = add_134 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_36: "f32[4, 1024, 1]" = var_mean_18[0]
        getitem_37: "f32[4, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_78: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_143, getitem_37);  add_143 = getitem_37 = None
        add_144: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_73: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_18);  sub_78 = rsqrt_18 = None
        mul_74: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg155_1);  mul_73 = arg155_1 = None
        add_145: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_74, arg156_1);  mul_74 = arg156_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_716: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_145, [4096, 768])
        permute_628: "f32[768, 3072]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        
        # No stacktrace found for following nodes
        mm_default_5: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_716, permute_628);  view_716 = permute_628 = None
        add_tensor_5: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_5, arg158_1);  mm_default_5 = arg158_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_717: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_5, [4, 1024, 3072]);  add_tensor_5 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_75: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_717, 0.5)
        mul_76: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_717, 0.7071067811865476);  view_717 = None
        erf_9: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_146: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_77: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_146);  mul_75 = add_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_718: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_77, [4096, 3072]);  mul_77 = None
        permute_629: "f32[3072, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        
        # No stacktrace found for following nodes
        mm_default_4: "f32[4096, 768]" = torch.ops.aten.mm.default(view_718, permute_629);  view_718 = permute_629 = None
        add_tensor_4: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_4, arg160_1);  mm_default_4 = arg160_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_719: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_4, [4, 1024, 768]);  add_tensor_4 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_147: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_719, add_145);  view_719 = add_145 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_38: "f32[4, 1024, 1]" = var_mean_19[0]
        getitem_39: "f32[4, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_205: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2)
        unsqueeze_206: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 3);  unsqueeze_205 = None
        full_default_109: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_90: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2320: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1);  slice_2320 = None
        slice_2316: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1)
        slice_2318: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2316, 3, 256, 9223372036854775807);  slice_2316 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_79: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_147, getitem_39);  add_147 = getitem_39 = None
        add_148: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_78: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_19);  sub_79 = rsqrt_19 = None
        mul_79: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg161_1);  mul_78 = arg161_1 = None
        add_149: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_79, arg162_1);  mul_79 = arg162_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_630: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_149, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_130: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format)
        view_720: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_130, [4096, 768]);  clone_130 = None
        permute_631: "f32[768, 768]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_40: "f32[4096, 768]" = torch.ops.aten.mm.default(view_720, permute_631);  view_720 = permute_631 = None
        view_721: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_40, [1024, 4, 768]);  mm_40 = None
        add_150: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_721, arg164_1);  view_721 = arg164_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_100: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_150, 8.0);  add_150 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_732: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_100, [1024, 4, 12, 64]);  div_100 = None
        permute_641: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_732, [1, 0, 2, 3]);  view_732 = None
        permute_642: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_641, [0, 2, 1, 3]);  permute_641 = None
        view_733: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_642, [48, 1024, 64]);  permute_642 = None
        view_734: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_733, [48, 2, 512, 64]);  view_733 = None
        as_strided_62: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_734, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_734 = None
        unsqueeze_192: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_62, 4);  as_strided_62 = None
        permute_643: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_192, [0, 1, 2, 4, 3]);  unsqueeze_192 = None
        permute_644: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_643, [0, 1, 2, 4, 3]);  permute_643 = None
        clone_133: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
        view_735: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_133, [144, 512, 64]);  clone_133 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_131: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format)
        view_722: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_131, [4096, 768]);  clone_131 = None
        permute_632: "f32[768, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_41: "f32[4096, 768]" = torch.ops.aten.mm.default(view_722, permute_632);  view_722 = permute_632 = None
        view_723: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_41, [1024, 4, 768]);  mm_41 = None
        add_151: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_723, arg166_1);  view_723 = arg166_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_727: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_151, [1024, 4, 12, 64]);  add_151 = None
        permute_635: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_727, [1, 0, 2, 3]);  view_727 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_637: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_635, [0, 2, 1, 3]);  permute_635 = None
        view_729: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_637, [48, 1024, 64]);  permute_637 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_731: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_729, [48, 2, 512, 64]);  view_729 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_61: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_731, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_731 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_191: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_61, 4);  as_strided_61 = None
        permute_639: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 4, 2, 3]);  unsqueeze_191 = None
        permute_645: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_639, [0, 1, 4, 3, 2]);  permute_639 = None
        clone_134: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_645, memory_format = torch.contiguous_format);  permute_645 = None
        view_736: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_134, [144, 64, 512]);  clone_134 = None
        bmm_20: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_735, view_736);  view_735 = view_736 = None
        view_737: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_20, [48, 3, 512, 1, 512]);  bmm_20 = None
        permute_646: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_737, [0, 1, 2, 4, 3]);  view_737 = None
        view_738: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_646, [48, 3, 512, 512]);  permute_646 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_40: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_738, [0, 0, 0, 1], 0.0);  view_738 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_739: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_40, [48, 3, 512, 513]);  constant_pad_nd_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2313: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_739, 2, 0, 256)
        slice_2314: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2313, 3, 0, 257);  slice_2313 = None
        copy_120: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2318, slice_2314);  slice_2318 = slice_2314 = None
        
        # No stacktrace found for following nodes
        slice_tensor_100: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_90, 1, 0, -1)
        slice_scatter_default_220: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_100, copy_120, 3, 256, 9223372036854775807);  slice_tensor_100 = copy_120 = None
        slice_scatter_default_221: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_90, slice_scatter_default_220, 1, 0, -1);  full_90 = slice_scatter_default_220 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_203: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_221, 1, -1);  select_203 = None
        select_202: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_221, 1, -1)
        slice_2334: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_202, 2, 256, 9223372036854775807);  select_202 = None
        select_200: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_739, 1, -1)
        slice_2327: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_200, 1, 256, 9223372036854775807);  select_200 = None
        slice_2328: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2327, 2, 0, 257);  slice_2327 = None
        copy_121: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_2334, slice_2328);  slice_2334 = slice_2328 = None
        
        # No stacktrace found for following nodes
        select_int_40: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_221, 1, -1)
        slice_scatter_default_222: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_40, copy_121, 2, 256, 9223372036854775807);  select_int_40 = copy_121 = None
        select_scatter_default_40: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_221, slice_scatter_default_222, 1, -1);  slice_scatter_default_221 = slice_scatter_default_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2353: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_40, 1, 1, 9223372036854775807);  slice_2353 = None
        slice_2349: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_40, 1, 1, 9223372036854775807)
        slice_2351: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2349, 3, 0, 256);  slice_2349 = None
        slice_2342: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_739, 2, -257, -1)
        slice_2343: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2342, 3, 257, 9223372036854775807);  slice_2342 = None
        copy_122: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2351, slice_2343);  slice_2351 = slice_2343 = None
        
        # No stacktrace found for following nodes
        slice_tensor_101: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_40, 1, 1, 9223372036854775807)
        slice_scatter_default_223: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_101, copy_122, 3, 0, 256);  slice_tensor_101 = copy_122 = None
        slice_scatter_default_224: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_40, slice_scatter_default_223, 1, 1, 9223372036854775807);  select_scatter_default_40 = slice_scatter_default_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_208: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_224, 1, 0)
        slice_2369: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_208, 1, 1, 256);  select_208 = slice_2369 = None
        select_207: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_224, 1, 0)
        slice_2366: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_207, 1, 1, 256);  select_207 = None
        slice_2367: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2366, 2, 1, 256);  slice_2366 = None
        select_205: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_739, 1, 0);  view_739 = None
        slice_2360: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_205, 1, 0, 255);  select_205 = None
        slice_2361: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2360, 2, -255, 9223372036854775807);  slice_2360 = None
        copy_123: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_2367, slice_2361);  slice_2367 = slice_2361 = None
        
        # No stacktrace found for following nodes
        select_int_41: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_224, 1, 0)
        slice_tensor_102: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_41, 1, 1, 256)
        slice_scatter_default_225: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_102, copy_123, 2, 1, 256);  slice_tensor_102 = copy_123 = None
        slice_scatter_default_226: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_41, slice_scatter_default_225, 1, 1, 256);  select_int_41 = slice_scatter_default_225 = None
        select_scatter_default_41: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_224, slice_scatter_default_226, 1, 0);  slice_scatter_default_224 = slice_scatter_default_226 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_744: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_41, [4, 12, 1024, 513])
        permute_651: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_744, [0, 2, 1, 3]);  view_744 = None
        slice_2392: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_651, 1, 0, 256);  slice_2392 = None
        view_743: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_41, [4, 12, 1024, 513])
        permute_650: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None
        slice_2388: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_650, 1, 0, 256);  permute_650 = None
        slice_2390: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2388, 3, 0, 257);  slice_2388 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_40: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_193: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_40, -2);  iota_40 = None
        iota_41: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_194: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        sub_81: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_193, unsqueeze_194);  unsqueeze_193 = unsqueeze_194 = None
        le_20: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_81, 0);  sub_81 = None
        full_default_100: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_101: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_80: "f32[256, 257]" = torch.ops.aten.where.self(le_20, full_default_100, full_default_101);  le_20 = full_default_100 = full_default_101 = None
        rev_40: "f32[256, 257]" = torch.ops.prims.rev.default(where_80, [0]);  where_80 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_195: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_40, 0);  rev_40 = None
        unsqueeze_196: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_40: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_196, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_50: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_40, torch.bool);  expand_40 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_102: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_742: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_41, [4, 12, 1024, 513]);  select_scatter_default_41 = None
        permute_649: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
        slice_2380: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_649, 1, 0, 256);  permute_649 = None
        slice_2382: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2380, 3, 0, 257);  slice_2380 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_81: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_50, full_default_102, slice_2382);  convert_element_type_50 = full_default_102 = slice_2382 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_124: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2390, where_81);  slice_2390 = where_81 = None
        
        # No stacktrace found for following nodes
        slice_tensor_103: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_651, 1, 0, 256)
        slice_scatter_default_227: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_103, copy_124, 3, 0, 257);  slice_tensor_103 = copy_124 = None
        slice_scatter_default_228: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_651, slice_scatter_default_227, 1, 0, 256);  permute_651 = slice_scatter_default_227 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_652: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_228, [0, 2, 1, 3]);  slice_scatter_default_228 = None
        view_745: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_652, [48, 4, 256, 513]);  permute_652 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_749: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_745, [4, 12, 1024, 513])
        permute_656: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_749, [0, 2, 1, 3]);  view_749 = None
        slice_2415: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_656, 1, -256, 9223372036854775807);  slice_2415 = None
        view_748: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_745, [4, 12, 1024, 513])
        permute_655: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
        slice_2411: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_655, 1, -256, 9223372036854775807);  permute_655 = None
        slice_2413: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2411, 3, -257, 9223372036854775807);  slice_2411 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_41: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_196, [1, 3]);  unsqueeze_196 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_41: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_41, [4, 256, 12, 257]);  rev_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_51: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_41, torch.bool);  expand_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_103: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_747: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_745, [4, 12, 1024, 513]);  view_745 = None
        permute_654: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_747, [0, 2, 1, 3]);  view_747 = None
        slice_2403: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_654, 1, -256, 9223372036854775807);  permute_654 = None
        slice_2405: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2403, 3, -257, 9223372036854775807);  slice_2403 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_82: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_51, full_default_103, slice_2405);  convert_element_type_51 = full_default_103 = slice_2405 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_125: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2413, where_82);  slice_2413 = where_82 = None
        
        # No stacktrace found for following nodes
        slice_tensor_104: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_656, 1, -256, 9223372036854775807)
        slice_scatter_default_229: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_104, copy_125, 3, -257, 9223372036854775807);  slice_tensor_104 = copy_125 = None
        slice_scatter_default_230: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_656, slice_scatter_default_229, 1, -256, 9223372036854775807);  permute_656 = slice_scatter_default_229 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_657: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_230, [0, 2, 1, 3]);  slice_scatter_default_230 = None
        view_750: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_657, [48, 4, 256, 513]);  permute_657 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_770: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_750, [4, 12, 1024, 513]);  view_750 = None
        permute_675: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_770, [0, 2, 1, 3]);  view_770 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_95: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2432: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1);  slice_2432 = None
        slice_2428: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1)
        slice_2430: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2428, 3, 256, 9223372036854775807);  slice_2428 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_94: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_659: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_94, [0, 2, 1, 3]);  full_94 = None
        view_752: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_659, [4, 1024, 1]);  permute_659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_754: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_752, [4, 2, 512, 1]);  view_752 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_63: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_754, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_754 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_199: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_63, 4);  as_strided_63 = None
        permute_661: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_199, [0, 1, 2, 4, 3]);  unsqueeze_199 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_104: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_10: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0)
        unsqueeze_197: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_10, 2);  ne_10 = None
        unsqueeze_198: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_52: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_198, torch.float32)
        where_83: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_198, full_default_104, convert_element_type_52);  unsqueeze_198 = full_default_104 = convert_element_type_52 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_660: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_83, [0, 2, 1, 3]);  where_83 = None
        view_753: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_660, [4, 1024, 1]);  permute_660 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_755: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_753, [4, 2, 512, 1]);  view_753 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_64: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_755, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_755 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_200: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_64, 4);  as_strided_64 = None
        permute_662: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 4, 2, 3]);  unsqueeze_200 = None
        mul_80: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_661, permute_662);  permute_661 = permute_662 = None
        view_756: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_80, [4, 3, 512, 512]);  mul_80 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_41: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_756, [0, 0, 0, 1], 0.0);  view_756 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_757: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_41, [4, 3, 512, 513]);  constant_pad_nd_41 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2425: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_757, 2, 0, 256)
        slice_2426: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2425, 3, 0, 257);  slice_2425 = None
        copy_126: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2430, slice_2426);  slice_2430 = slice_2426 = None
        
        # No stacktrace found for following nodes
        slice_tensor_105: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_95, 1, 0, -1)
        slice_scatter_default_231: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_105, copy_126, 3, 256, 9223372036854775807);  slice_tensor_105 = copy_126 = None
        slice_scatter_default_232: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_95, slice_scatter_default_231, 1, 0, -1);  full_95 = slice_scatter_default_231 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_213: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_232, 1, -1);  select_213 = None
        select_212: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_232, 1, -1)
        slice_2446: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_212, 2, 256, 9223372036854775807);  select_212 = None
        select_210: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_757, 1, -1)
        slice_2439: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_210, 1, 256, 9223372036854775807);  select_210 = None
        slice_2440: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2439, 2, 0, 257);  slice_2439 = None
        copy_127: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_2446, slice_2440);  slice_2446 = slice_2440 = None
        
        # No stacktrace found for following nodes
        select_int_42: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_232, 1, -1)
        slice_scatter_default_233: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_42, copy_127, 2, 256, 9223372036854775807);  select_int_42 = copy_127 = None
        select_scatter_default_42: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_232, slice_scatter_default_233, 1, -1);  slice_scatter_default_232 = slice_scatter_default_233 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2465: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_42, 1, 1, 9223372036854775807);  slice_2465 = None
        slice_2461: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_42, 1, 1, 9223372036854775807)
        slice_2463: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2461, 3, 0, 256);  slice_2461 = None
        slice_2454: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_757, 2, -257, -1)
        slice_2455: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2454, 3, 257, 9223372036854775807);  slice_2454 = None
        copy_128: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2463, slice_2455);  slice_2463 = slice_2455 = None
        
        # No stacktrace found for following nodes
        slice_tensor_106: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_42, 1, 1, 9223372036854775807)
        slice_scatter_default_234: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_106, copy_128, 3, 0, 256);  slice_tensor_106 = copy_128 = None
        slice_scatter_default_235: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_42, slice_scatter_default_234, 1, 1, 9223372036854775807);  select_scatter_default_42 = slice_scatter_default_234 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_218: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_235, 1, 0)
        slice_2481: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_218, 1, 1, 256);  select_218 = slice_2481 = None
        select_217: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_235, 1, 0)
        slice_2478: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_217, 1, 1, 256);  select_217 = None
        slice_2479: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2478, 2, 1, 256);  slice_2478 = None
        select_215: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_757, 1, 0);  view_757 = None
        slice_2472: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_215, 1, 0, 255);  select_215 = None
        slice_2473: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2472, 2, -255, 9223372036854775807);  slice_2472 = None
        copy_129: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_2479, slice_2473);  slice_2479 = slice_2473 = None
        
        # No stacktrace found for following nodes
        select_int_43: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_235, 1, 0)
        slice_tensor_107: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_43, 1, 1, 256)
        slice_scatter_default_236: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_107, copy_129, 2, 1, 256);  slice_tensor_107 = copy_129 = None
        slice_scatter_default_237: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_43, slice_scatter_default_236, 1, 1, 256);  select_int_43 = slice_scatter_default_236 = None
        select_scatter_default_43: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_235, slice_scatter_default_237, 1, 0);  slice_scatter_default_235 = slice_scatter_default_237 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_762: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_43, [4, 1, 1024, 513])
        permute_667: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_762, [0, 2, 1, 3]);  view_762 = None
        slice_2504: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_667, 1, 0, 256);  slice_2504 = None
        view_761: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_43, [4, 1, 1024, 513])
        permute_666: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_761, [0, 2, 1, 3]);  view_761 = None
        slice_2500: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_666, 1, 0, 256);  permute_666 = None
        slice_2502: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2500, 3, 0, 257);  slice_2500 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_42: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_201: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_42, -2);  iota_42 = None
        iota_43: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_202: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
        sub_83: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_201, unsqueeze_202);  unsqueeze_201 = unsqueeze_202 = None
        le_21: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_83, 0);  sub_83 = None
        full_default_105: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_84: "f32[256, 257]" = torch.ops.aten.where.self(le_21, full_default_105, full_default_106);  le_21 = full_default_105 = full_default_106 = None
        rev_42: "f32[256, 257]" = torch.ops.prims.rev.default(where_84, [0]);  where_84 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_203: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_42, 0);  rev_42 = None
        unsqueeze_204: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_42: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_204, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_53: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_42, torch.bool);  expand_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_107: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_760: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_43, [4, 1, 1024, 513]);  select_scatter_default_43 = None
        permute_665: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None
        slice_2492: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_665, 1, 0, 256);  permute_665 = None
        slice_2494: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2492, 3, 0, 257);  slice_2492 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_85: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_53, full_default_107, slice_2494);  convert_element_type_53 = full_default_107 = slice_2494 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_130: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2502, where_85);  slice_2502 = where_85 = None
        
        # No stacktrace found for following nodes
        slice_tensor_108: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_667, 1, 0, 256)
        slice_scatter_default_238: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_108, copy_130, 3, 0, 257);  slice_tensor_108 = copy_130 = None
        slice_scatter_default_239: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_667, slice_scatter_default_238, 1, 0, 256);  permute_667 = slice_scatter_default_238 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_668: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_239, [0, 2, 1, 3]);  slice_scatter_default_239 = None
        view_763: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_668, [4, 4, 256, 513]);  permute_668 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_767: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_763, [4, 1, 1024, 513])
        permute_672: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_767, [0, 2, 1, 3]);  view_767 = None
        slice_2527: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_672, 1, -256, 9223372036854775807);  slice_2527 = None
        view_766: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_763, [4, 1, 1024, 513])
        permute_671: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
        slice_2523: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_671, 1, -256, 9223372036854775807);  permute_671 = None
        slice_2525: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2523, 3, -257, 9223372036854775807);  slice_2523 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_43: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_204, [1, 3]);  unsqueeze_204 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_43: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_43, [4, 256, 1, 257]);  rev_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_54: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_43, torch.bool);  expand_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_108: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_765: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_763, [4, 1, 1024, 513]);  view_763 = None
        permute_670: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_765, [0, 2, 1, 3]);  view_765 = None
        slice_2515: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_670, 1, -256, 9223372036854775807);  permute_670 = None
        slice_2517: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2515, 3, -257, 9223372036854775807);  slice_2515 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_86: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_54, full_default_108, slice_2517);  convert_element_type_54 = full_default_108 = slice_2517 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_131: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2525, where_86);  slice_2525 = where_86 = None
        
        # No stacktrace found for following nodes
        slice_tensor_109: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_672, 1, -256, 9223372036854775807)
        slice_scatter_default_240: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_109, copy_131, 3, -257, 9223372036854775807);  slice_tensor_109 = copy_131 = None
        slice_scatter_default_241: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_672, slice_scatter_default_240, 1, -256, 9223372036854775807);  permute_672 = slice_scatter_default_240 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_673: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_241, [0, 2, 1, 3]);  slice_scatter_default_241 = None
        view_768: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_673, [4, 4, 256, 513]);  permute_673 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_771: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_768, [4, 1, 1024, 513]);  view_768 = None
        permute_676: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None
        add_155: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_675, permute_676);  permute_675 = permute_676 = None
        permute_677: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_155, [0, 2, 1, 3]);  add_155 = None
        view_773: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_677, [48, 4, 256, 513]);  permute_677 = None
        view_774: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_773, [4, 12, 1024, 513]);  view_773 = None
        permute_678: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_135: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_678, memory_format = torch.contiguous_format);  permute_678 = None
        amax_10: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_135, [-1], True)
        sub_84: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_135, amax_10);  clone_135 = amax_10 = None
        exp_10: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
        sum_11: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_107: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_87: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_206, full_default_109, div_107);  unsqueeze_206 = full_default_109 = div_107 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_680: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_87, [0, 2, 1, 3]);  where_87 = None
        clone_137: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_680, memory_format = torch.contiguous_format);  permute_680 = None
        view_776: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_137, [48, 4, 256, 513]);  clone_137 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_43: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_776, [0, 257], 0.0);  view_776 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_778: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_43, [48, 4, -1]);  constant_pad_nd_43 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_2537: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_778, 2, 0, -256);  view_778 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_779: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_2537, [48, 4, 256, 769]);  slice_2537 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_2541: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_779, 3, 0, -1);  view_779 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_207: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2541, 4);  slice_2541 = None
        permute_682: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_207, [0, 1, 2, 4, 3]);  unsqueeze_207 = None
        permute_684: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_682, [0, 1, 2, 4, 3]);  permute_682 = None
        view_780: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_684, [192, 256, 768]);  permute_684 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_132: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
        view_724: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_132, [4096, 768]);  clone_132 = None
        permute_633: "f32[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        mm_42: "f32[4096, 768]" = torch.ops.aten.mm.default(view_724, permute_633);  view_724 = permute_633 = None
        view_725: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_42, [1024, 4, 768]);  mm_42 = None
        add_152: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_725, arg168_1);  view_725 = arg168_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_775: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_152, [1024, 4, 12, 64]);  add_152 = None
        permute_679: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_775, [1, 0, 2, 3]);  view_775 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_681: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_679, [0, 2, 1, 3]);  permute_679 = None
        view_777: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_681, [48, 1024, 64]);  permute_681 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_42: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_777, [0, 0, 256, 256], -1.0);  view_777 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_65: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_42, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_42 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_208: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_65, 4);  as_strided_65 = None
        permute_683: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_208, [0, 1, 4, 3, 2]);  unsqueeze_208 = None
        permute_685: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_683, [0, 1, 4, 3, 2]);  permute_683 = None
        clone_138: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
        view_781: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_138, [192, 768, 64]);  clone_138 = None
        bmm_21: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_780, view_781);  view_780 = view_781 = None
        view_782: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_21, [48, 4, 256, 1, 64]);  bmm_21 = None
        permute_686: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_782, [0, 1, 2, 4, 3]);  view_782 = None
        view_783: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_686, [48, 4, 256, 64]);  permute_686 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_784: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_783, [4, 12, 1024, 64]);  view_783 = None
        permute_687: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_688: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_687, [1, 0, 2, 3]);  permute_687 = None
        clone_139: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
        view_785: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_139, [1024, 4, 768]);  clone_139 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_689: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_785, [1, 0, 2]);  view_785 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_140: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_689, memory_format = torch.contiguous_format);  permute_689 = None
        view_786: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_140, [4096, 768]);  clone_140 = None
        permute_690: "f32[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_43: "f32[4096, 768]" = torch.ops.aten.mm.default(view_786, permute_690);  view_786 = permute_690 = None
        view_787: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_43, [4, 1024, 768]);  mm_43 = None
        add_157: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_787, arg170_1);  view_787 = arg170_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_158: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_157, add_149);  add_157 = add_149 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_40: "f32[4, 1024, 1]" = var_mean_20[0]
        getitem_41: "f32[4, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_86: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_158, getitem_41);  add_158 = getitem_41 = None
        add_159: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_81: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_20);  sub_86 = rsqrt_20 = None
        mul_82: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg171_1);  mul_81 = arg171_1 = None
        add_160: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_82, arg172_1);  mul_82 = arg172_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_788: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_160, [4096, 768])
        permute_691: "f32[768, 3072]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        
        # No stacktrace found for following nodes
        mm_default_3: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_788, permute_691);  view_788 = permute_691 = None
        add_tensor_3: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_3, arg174_1);  mm_default_3 = arg174_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_789: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_3, [4, 1024, 3072]);  add_tensor_3 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_83: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_789, 0.5)
        mul_84: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_789, 0.7071067811865476);  view_789 = None
        erf_10: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_161: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_85: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_83, add_161);  mul_83 = add_161 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_790: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_85, [4096, 3072]);  mul_85 = None
        permute_692: "f32[3072, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        
        # No stacktrace found for following nodes
        mm_default_2: "f32[4096, 768]" = torch.ops.aten.mm.default(view_790, permute_692);  view_790 = permute_692 = None
        add_tensor_2: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default_2, arg176_1);  mm_default_2 = arg176_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_791: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [4, 1024, 768]);  add_tensor_2 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_162: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_791, add_160);  view_791 = add_160 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
        getitem_42: "f32[4, 1024, 1]" = var_mean_21[0]
        getitem_43: "f32[4, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_224: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2);  arg8_1 = None
        unsqueeze_225: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        full_default_119: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_99: "f32[48, 4, 256, 513]" = torch.ops.aten.full.default([48, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2551: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1);  slice_2551 = None
        slice_2547: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1)
        slice_2549: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2547, 3, 256, 9223372036854775807);  slice_2547 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        sub_87: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_162, getitem_43);  add_162 = getitem_43 = None
        add_163: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_86: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_21);  sub_87 = rsqrt_21 = None
        mul_87: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_86, arg177_1);  mul_86 = arg177_1 = None
        add_164: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_87, arg178_1);  mul_87 = arg178_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:544 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_693: "f32[1024, 4, 768]" = torch.ops.aten.permute.default(add_164, [1, 0, 2])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:547 in forward, code: query_vectors = self.query(hidden_states)
        clone_143: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format)
        view_792: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_143, [4096, 768]);  clone_143 = None
        permute_694: "f32[768, 768]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        mm_44: "f32[4096, 768]" = torch.ops.aten.mm.default(view_792, permute_694);  view_792 = permute_694 = None
        view_793: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_44, [1024, 4, 768]);  mm_44 = None
        add_165: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_793, arg180_1);  view_793 = arg180_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_110: "f32[1024, 4, 768]" = torch.ops.aten.div.Tensor(add_165, 8.0);  add_165 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_804: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(div_110, [1024, 4, 12, 64]);  div_110 = None
        permute_704: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_804, [1, 0, 2, 3]);  view_804 = None
        permute_705: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_704, [0, 2, 1, 3]);  permute_704 = None
        view_805: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_705, [48, 1024, 64]);  permute_705 = None
        view_806: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_805, [48, 2, 512, 64]);  view_805 = None
        as_strided_68: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_806, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_806 = None
        unsqueeze_211: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_68, 4);  as_strided_68 = None
        permute_706: "f32[48, 3, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_211, [0, 1, 2, 4, 3]);  unsqueeze_211 = None
        permute_707: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_706, [0, 1, 2, 4, 3]);  permute_706 = None
        clone_146: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
        view_807: "f32[144, 512, 64]" = torch.ops.aten.reshape.default(clone_146, [144, 512, 64]);  clone_146 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:548 in forward, code: key_vectors = self.key(hidden_states)
        clone_144: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format)
        view_794: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_144, [4096, 768]);  clone_144 = None
        permute_695: "f32[768, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_45: "f32[4096, 768]" = torch.ops.aten.mm.default(view_794, permute_695);  view_794 = permute_695 = None
        view_795: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_45, [1024, 4, 768]);  mm_45 = None
        add_166: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_795, arg182_1);  view_795 = arg182_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:560 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_799: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_166, [1024, 4, 12, 64]);  add_166 = None
        permute_698: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_799, [1, 0, 2, 3]);  view_799 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_700: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_698, [0, 2, 1, 3]);  permute_698 = None
        view_801: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_700, [48, 1024, 64]);  permute_700 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_803: "f32[48, 2, 512, 64]" = torch.ops.aten.reshape.default(view_801, [48, 2, 512, 64]);  view_801 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_67: "f32[48, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_803, [48, 3, 512, 64], [64, 786432, 3072, 1]);  view_803 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_210: "f32[48, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_67, 4);  as_strided_67 = None
        permute_702: "f32[48, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_210, [0, 1, 4, 2, 3]);  unsqueeze_210 = None
        permute_708: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_702, [0, 1, 4, 3, 2]);  permute_702 = None
        clone_147: "f32[48, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_708, memory_format = torch.contiguous_format);  permute_708 = None
        view_808: "f32[144, 64, 512]" = torch.ops.aten.reshape.default(clone_147, [144, 64, 512]);  clone_147 = None
        bmm_22: "f32[144, 512, 512]" = torch.ops.aten.bmm.default(view_807, view_808);  view_807 = view_808 = None
        view_809: "f32[48, 3, 512, 1, 512]" = torch.ops.aten.reshape.default(bmm_22, [48, 3, 512, 1, 512]);  bmm_22 = None
        permute_709: "f32[48, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_809, [0, 1, 2, 4, 3]);  view_809 = None
        view_810: "f32[48, 3, 512, 512]" = torch.ops.aten.reshape.default(permute_709, [48, 3, 512, 512]);  permute_709 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_44: "f32[48, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_810, [0, 0, 0, 1], 0.0);  view_810 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_811: "f32[48, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_44, [48, 3, 512, 513]);  constant_pad_nd_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2544: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_811, 2, 0, 256)
        slice_2545: "f32[48, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2544, 3, 0, 257);  slice_2544 = None
        copy_132: "f32[48, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2549, slice_2545);  slice_2549 = slice_2545 = None
        
        # No stacktrace found for following nodes
        slice_tensor_110: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_99, 1, 0, -1)
        slice_scatter_default_242: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_110, copy_132, 3, 256, 9223372036854775807);  slice_tensor_110 = copy_132 = None
        slice_scatter_default_243: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_99, slice_scatter_default_242, 1, 0, -1);  full_99 = slice_scatter_default_242 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_223: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_243, 1, -1);  select_223 = None
        select_222: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_243, 1, -1)
        slice_2565: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(select_222, 2, 256, 9223372036854775807);  select_222 = None
        select_220: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_811, 1, -1)
        slice_2558: "f32[48, 256, 513]" = torch.ops.aten.slice.Tensor(select_220, 1, 256, 9223372036854775807);  select_220 = None
        slice_2559: "f32[48, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2558, 2, 0, 257);  slice_2558 = None
        copy_133: "f32[48, 256, 257]" = torch.ops.aten.copy.default(slice_2565, slice_2559);  slice_2565 = slice_2559 = None
        
        # No stacktrace found for following nodes
        select_int_44: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_243, 1, -1)
        slice_scatter_default_244: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_44, copy_133, 2, 256, 9223372036854775807);  select_int_44 = copy_133 = None
        select_scatter_default_44: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_243, slice_scatter_default_244, 1, -1);  slice_scatter_default_243 = slice_scatter_default_244 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2584: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_44, 1, 1, 9223372036854775807);  slice_2584 = None
        slice_2580: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_44, 1, 1, 9223372036854775807)
        slice_2582: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2580, 3, 0, 256);  slice_2580 = None
        slice_2573: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_811, 2, -257, -1)
        slice_2574: "f32[48, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2573, 3, 257, 9223372036854775807);  slice_2573 = None
        copy_134: "f32[48, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2582, slice_2574);  slice_2582 = slice_2574 = None
        
        # No stacktrace found for following nodes
        slice_tensor_111: "f32[48, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_44, 1, 1, 9223372036854775807)
        slice_scatter_default_245: "f32[48, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_111, copy_134, 3, 0, 256);  slice_tensor_111 = copy_134 = None
        slice_scatter_default_246: "f32[48, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_44, slice_scatter_default_245, 1, 1, 9223372036854775807);  select_scatter_default_44 = slice_scatter_default_245 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_228: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_246, 1, 0)
        slice_2600: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_228, 1, 1, 256);  select_228 = slice_2600 = None
        select_227: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_246, 1, 0)
        slice_2597: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_227, 1, 1, 256);  select_227 = None
        slice_2598: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2597, 2, 1, 256);  slice_2597 = None
        select_225: "f32[48, 512, 513]" = torch.ops.aten.select.int(view_811, 1, 0);  view_811 = None
        slice_2591: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_225, 1, 0, 255);  select_225 = None
        slice_2592: "f32[48, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2591, 2, -255, 9223372036854775807);  slice_2591 = None
        copy_135: "f32[48, 255, 255]" = torch.ops.aten.copy.default(slice_2598, slice_2592);  slice_2598 = slice_2592 = None
        
        # No stacktrace found for following nodes
        select_int_45: "f32[48, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_246, 1, 0)
        slice_tensor_112: "f32[48, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_45, 1, 1, 256)
        slice_scatter_default_247: "f32[48, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_112, copy_135, 2, 1, 256);  slice_tensor_112 = copy_135 = None
        slice_scatter_default_248: "f32[48, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_45, slice_scatter_default_247, 1, 1, 256);  select_int_45 = slice_scatter_default_247 = None
        select_scatter_default_45: "f32[48, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_246, slice_scatter_default_248, 1, 0);  slice_scatter_default_246 = slice_scatter_default_248 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_816: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_45, [4, 12, 1024, 513])
        permute_714: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_816, [0, 2, 1, 3]);  view_816 = None
        slice_2623: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_714, 1, 0, 256);  slice_2623 = None
        view_815: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_45, [4, 12, 1024, 513])
        permute_713: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_815, [0, 2, 1, 3]);  view_815 = None
        slice_2619: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_713, 1, 0, 256);  permute_713 = None
        slice_2621: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2619, 3, 0, 257);  slice_2619 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_44: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_212: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_44, -2);  iota_44 = None
        iota_45: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_213: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        sub_89: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_212, unsqueeze_213);  unsqueeze_212 = unsqueeze_213 = None
        le_22: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_89, 0);  sub_89 = None
        full_default_110: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_111: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_88: "f32[256, 257]" = torch.ops.aten.where.self(le_22, full_default_110, full_default_111);  le_22 = full_default_110 = full_default_111 = None
        rev_44: "f32[256, 257]" = torch.ops.prims.rev.default(where_88, [0]);  where_88 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_214: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_44, 0);  rev_44 = None
        unsqueeze_215: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_44: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_215, [4, 256, 12, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_55: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_44, torch.bool);  expand_44 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_112: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_814: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_45, [4, 12, 1024, 513]);  select_scatter_default_45 = None
        permute_712: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None
        slice_2611: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_712, 1, 0, 256);  permute_712 = None
        slice_2613: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2611, 3, 0, 257);  slice_2611 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_89: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_55, full_default_112, slice_2613);  convert_element_type_55 = full_default_112 = slice_2613 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_136: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2621, where_89);  slice_2621 = where_89 = None
        
        # No stacktrace found for following nodes
        slice_tensor_113: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_714, 1, 0, 256)
        slice_scatter_default_249: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_113, copy_136, 3, 0, 257);  slice_tensor_113 = copy_136 = None
        slice_scatter_default_250: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_714, slice_scatter_default_249, 1, 0, 256);  permute_714 = slice_scatter_default_249 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_715: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_250, [0, 2, 1, 3]);  slice_scatter_default_250 = None
        view_817: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_715, [48, 4, 256, 513]);  permute_715 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_821: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_817, [4, 12, 1024, 513])
        permute_719: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_821, [0, 2, 1, 3]);  view_821 = None
        slice_2646: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_719, 1, -256, 9223372036854775807);  slice_2646 = None
        view_820: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_817, [4, 12, 1024, 513])
        permute_718: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_820, [0, 2, 1, 3]);  view_820 = None
        slice_2642: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_718, 1, -256, 9223372036854775807);  permute_718 = None
        slice_2644: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2642, 3, -257, 9223372036854775807);  slice_2642 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_45: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_215, [1, 3]);  unsqueeze_215 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_45: "f32[4, 256, 12, 257]" = torch.ops.aten.expand.default(rev_45, [4, 256, 12, 257]);  rev_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_56: "b8[4, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_45, torch.bool);  expand_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_113: "f32[4, 256, 12, 257]" = torch.ops.aten.full.default([4, 256, 12, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_819: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_817, [4, 12, 1024, 513]);  view_817 = None
        permute_717: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_819, [0, 2, 1, 3]);  view_819 = None
        slice_2634: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_717, 1, -256, 9223372036854775807);  permute_717 = None
        slice_2636: "f32[4, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2634, 3, -257, 9223372036854775807);  slice_2634 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_90: "f32[4, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_56, full_default_113, slice_2636);  convert_element_type_56 = full_default_113 = slice_2636 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_137: "f32[4, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2644, where_90);  slice_2644 = where_90 = None
        
        # No stacktrace found for following nodes
        slice_tensor_114: "f32[4, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_719, 1, -256, 9223372036854775807)
        slice_scatter_default_251: "f32[4, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_114, copy_137, 3, -257, 9223372036854775807);  slice_tensor_114 = copy_137 = None
        slice_scatter_default_252: "f32[4, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_719, slice_scatter_default_251, 1, -256, 9223372036854775807);  permute_719 = slice_scatter_default_251 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_720: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_252, [0, 2, 1, 3]);  slice_scatter_default_252 = None
        view_822: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_720, [48, 4, 256, 513]);  permute_720 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_842: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_822, [4, 12, 1024, 513]);  view_822 = None
        permute_738: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_842, [0, 2, 1, 3]);  view_842 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:846 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
        full_104: "f32[4, 4, 256, 513]" = torch.ops.aten.full.default([4, 4, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2663: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1);  slice_2663 = None
        slice_2659: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_2661: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2659, 3, 256, 9223372036854775807);  slice_2659 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:575 in forward, code: float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        full_103: "f32[4, 1024, 1, 1]" = torch.ops.aten.full.default([4, 1024, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:824 in _sliding_chunks_query_key_matmul, code: query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_722: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(full_103, [0, 2, 1, 3]);  full_103 = None
        view_824: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_722, [4, 1024, 1]);  permute_722 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_826: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_824, [4, 2, 512, 1]);  view_824 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_69: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_826, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_826 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_218: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_69, 4);  as_strided_69 = None
        permute_724: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_218, [0, 1, 2, 4, 3]);  unsqueeze_218 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        full_default_114: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:567 in forward, code: remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        ne_11: "b8[4, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0);  arg7_1 = None
        unsqueeze_216: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_11, 2);  ne_11 = None
        unsqueeze_217: "b8[4, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:570 in forward, code: float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
        convert_element_type_57: "f32[4, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_217, torch.float32)
        where_91: "f32[4, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_217, full_default_114, convert_element_type_57);  unsqueeze_217 = full_default_114 = convert_element_type_57 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:825 in _sliding_chunks_query_key_matmul, code: key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_723: "f32[4, 1, 1024, 1]" = torch.ops.aten.permute.default(where_91, [0, 2, 1, 3]);  where_91 = None
        view_825: "f32[4, 1024, 1]" = torch.ops.aten.reshape.default(permute_723, [4, 1024, 1]);  permute_723 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:757 in _chunk, code: hidden_states = hidden_states.view(
        view_827: "f32[4, 2, 512, 1]" = torch.ops.aten.reshape.default(view_825, [4, 2, 512, 1]);  view_825 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:769 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        as_strided_70: "f32[4, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_827, [4, 3, 512, 1], [1024, 256, 1, 1]);  view_827 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:834 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        unsqueeze_219: "f32[4, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_70, 4);  as_strided_70 = None
        permute_725: "f32[4, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_219, [0, 1, 4, 2, 3]);  unsqueeze_219 = None
        mul_88: "f32[4, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_724, permute_725);  permute_724 = permute_725 = None
        view_828: "f32[4, 3, 512, 512]" = torch.ops.aten.reshape.default(mul_88, [4, 3, 512, 512]);  mul_88 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_45: "f32[4, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_828, [0, 0, 0, 1], 0.0);  view_828 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_829: "f32[4, 3, 512, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_45, [4, 3, 512, 513]);  constant_pad_nd_45 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:852 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2656: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_829, 2, 0, 256)
        slice_2657: "f32[4, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2656, 3, 0, 257);  slice_2656 = None
        copy_138: "f32[4, 3, 256, 257]" = torch.ops.aten.copy.default(slice_2661, slice_2657);  slice_2661 = slice_2657 = None
        
        # No stacktrace found for following nodes
        slice_tensor_115: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_scatter_default_253: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_115, copy_138, 3, 256, 9223372036854775807);  slice_tensor_115 = copy_138 = None
        slice_scatter_default_254: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_104, slice_scatter_default_253, 1, 0, -1);  full_104 = slice_scatter_default_253 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:855 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_233: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_254, 1, -1);  select_233 = None
        select_232: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_254, 1, -1)
        slice_2677: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(select_232, 2, 256, 9223372036854775807);  select_232 = None
        select_230: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_829, 1, -1)
        slice_2670: "f32[4, 256, 513]" = torch.ops.aten.slice.Tensor(select_230, 1, 256, 9223372036854775807);  select_230 = None
        slice_2671: "f32[4, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2670, 2, 0, 257);  slice_2670 = None
        copy_139: "f32[4, 256, 257]" = torch.ops.aten.copy.default(slice_2677, slice_2671);  slice_2677 = slice_2671 = None
        
        # No stacktrace found for following nodes
        select_int_46: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_254, 1, -1)
        slice_scatter_default_255: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_46, copy_139, 2, 256, 9223372036854775807);  select_int_46 = copy_139 = None
        select_scatter_default_46: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_254, slice_scatter_default_255, 1, -1);  slice_scatter_default_254 = slice_scatter_default_255 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:859 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2696: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_46, 1, 1, 9223372036854775807);  slice_2696 = None
        slice_2692: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_46, 1, 1, 9223372036854775807)
        slice_2694: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2692, 3, 0, 256);  slice_2692 = None
        slice_2685: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_829, 2, -257, -1)
        slice_2686: "f32[4, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2685, 3, 257, 9223372036854775807);  slice_2685 = None
        copy_140: "f32[4, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2694, slice_2686);  slice_2694 = slice_2686 = None
        
        # No stacktrace found for following nodes
        slice_tensor_116: "f32[4, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_default_46, 1, 1, 9223372036854775807)
        slice_scatter_default_256: "f32[4, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_116, copy_140, 3, 0, 256);  slice_tensor_116 = copy_140 = None
        slice_scatter_default_257: "f32[4, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_default_46, slice_scatter_default_256, 1, 1, 9223372036854775807);  select_scatter_default_46 = slice_scatter_default_256 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:863 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_238: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_257, 1, 0)
        slice_2712: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_238, 1, 1, 256);  select_238 = slice_2712 = None
        select_237: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_257, 1, 0)
        slice_2709: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_237, 1, 1, 256);  select_237 = None
        slice_2710: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2709, 2, 1, 256);  slice_2709 = None
        select_235: "f32[4, 512, 513]" = torch.ops.aten.select.int(view_829, 1, 0);  view_829 = None
        slice_2703: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_235, 1, 0, 255);  select_235 = None
        slice_2704: "f32[4, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2703, 2, -255, 9223372036854775807);  slice_2703 = None
        copy_141: "f32[4, 255, 255]" = torch.ops.aten.copy.default(slice_2710, slice_2704);  slice_2710 = slice_2704 = None
        
        # No stacktrace found for following nodes
        select_int_47: "f32[4, 256, 513]" = torch.ops.aten.select.int(slice_scatter_default_257, 1, 0)
        slice_tensor_117: "f32[4, 255, 513]" = torch.ops.aten.slice.Tensor(select_int_47, 1, 1, 256)
        slice_scatter_default_258: "f32[4, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_117, copy_141, 2, 1, 256);  slice_tensor_117 = copy_141 = None
        slice_scatter_default_259: "f32[4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_int_47, slice_scatter_default_258, 1, 1, 256);  select_int_47 = slice_scatter_default_258 = None
        select_scatter_default_47: "f32[4, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_default_257, slice_scatter_default_259, 1, 0);  slice_scatter_default_257 = slice_scatter_default_259 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_834: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_47, [4, 1, 1024, 513])
        permute_730: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_834, [0, 2, 1, 3]);  view_834 = None
        slice_2735: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_730, 1, 0, 256);  slice_2735 = None
        view_833: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_47, [4, 1, 1024, 513])
        permute_729: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None
        slice_2731: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_729, 1, 0, 256);  permute_729 = None
        slice_2733: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2731, 3, 0, 257);  slice_2731 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:795 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        iota_46: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_220: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_46, -2);  iota_46 = None
        iota_47: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_221: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
        sub_91: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_220, unsqueeze_221);  unsqueeze_220 = unsqueeze_221 = None
        le_23: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_91, 0);  sub_91 = None
        full_default_115: "f32[256, 257]" = torch.ops.aten.full.default([256, 257], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_116: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_92: "f32[256, 257]" = torch.ops.aten.where.self(le_23, full_default_115, full_default_116);  le_23 = full_default_115 = full_default_116 = None
        rev_46: "f32[256, 257]" = torch.ops.prims.rev.default(where_92, [0]);  where_92 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:796 in _mask_invalid_locations, code: beginning_mask = beginning_mask_2d[None, :, None, :]
        unsqueeze_222: "f32[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_46, 0);  rev_46 = None
        unsqueeze_223: "f32[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:799 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand_46: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_223, [4, 256, 1, 257])
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_58: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_46, torch.bool);  expand_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        full_default_117: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_832: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(select_scatter_default_47, [4, 1, 1024, 513]);  select_scatter_default_47 = None
        permute_728: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_832, [0, 2, 1, 3]);  view_832 = None
        slice_2723: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_728, 1, 0, 256);  permute_728 = None
        slice_2725: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2723, 3, 0, 257);  slice_2723 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:802 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_93: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_58, full_default_117, slice_2725);  convert_element_type_58 = full_default_117 = slice_2725 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        copy_142: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2733, where_93);  slice_2733 = where_93 = None
        
        # No stacktrace found for following nodes
        slice_tensor_118: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_730, 1, 0, 256)
        slice_scatter_default_260: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_118, copy_142, 3, 0, 257);  slice_tensor_118 = copy_142 = None
        slice_scatter_default_261: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_730, slice_scatter_default_260, 1, 0, 256);  permute_730 = slice_scatter_default_260 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:800 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        permute_731: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_261, [0, 2, 1, 3]);  slice_scatter_default_261 = None
        view_835: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_731, [4, 4, 256, 513]);  permute_731 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_839: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_835, [4, 1, 1024, 513])
        permute_735: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_839, [0, 2, 1, 3]);  view_839 = None
        slice_2758: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_735, 1, -256, 9223372036854775807);  slice_2758 = None
        view_838: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_835, [4, 1, 1024, 513])
        permute_734: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_838, [0, 2, 1, 3]);  view_838 = None
        slice_2754: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_734, 1, -256, 9223372036854775807);  permute_734 = None
        slice_2756: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2754, 3, -257, 9223372036854775807);  slice_2754 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:797 in _mask_invalid_locations, code: ending_mask = beginning_mask.flip(dims=(1, 3))
        rev_47: "f32[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_223, [1, 3]);  unsqueeze_223 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_47: "f32[4, 256, 1, 257]" = torch.ops.aten.expand.default(rev_47, [4, 256, 1, 257]);  rev_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_59: "b8[4, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_47, torch.bool);  expand_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        full_default_118: "f32[4, 256, 1, 257]" = torch.ops.aten.full.default([4, 256, 1, 257], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_837: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_835, [4, 1, 1024, 513]);  view_835 = None
        permute_733: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_837, [0, 2, 1, 3]);  view_837 = None
        slice_2746: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_733, 1, -256, 9223372036854775807);  permute_733 = None
        slice_2748: "f32[4, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_2746, 3, -257, 9223372036854775807);  slice_2746 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_94: "f32[4, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_59, full_default_118, slice_2748);  convert_element_type_59 = full_default_118 = slice_2748 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        copy_143: "f32[4, 256, 1, 257]" = torch.ops.aten.copy.default(slice_2756, where_94);  slice_2756 = where_94 = None
        
        # No stacktrace found for following nodes
        slice_tensor_119: "f32[4, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_735, 1, -256, 9223372036854775807)
        slice_scatter_default_262: "f32[4, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_tensor_119, copy_143, 3, -257, 9223372036854775807);  slice_tensor_119 = copy_143 = None
        slice_scatter_default_263: "f32[4, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_735, slice_scatter_default_262, 1, -256, 9223372036854775807);  permute_735 = slice_scatter_default_262 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:805 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_736: "f32[4, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_default_263, [0, 2, 1, 3]);  slice_scatter_default_263 = None
        view_840: "f32[4, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_736, [4, 4, 256, 513]);  permute_736 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_scores += diagonal_mask
        view_843: "f32[4, 1, 1024, 513]" = torch.ops.aten.reshape.default(view_840, [4, 1, 1024, 513]);  view_840 = None
        permute_739: "f32[4, 1024, 1, 513]" = torch.ops.aten.permute.default(view_843, [0, 2, 1, 3]);  view_843 = None
        add_170: "f32[4, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_738, permute_739);  permute_738 = permute_739 = None
        permute_740: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(add_170, [0, 2, 1, 3]);  add_170 = None
        view_845: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_740, [48, 4, 256, 513]);  permute_740 = None
        view_846: "f32[4, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_845, [4, 12, 1024, 513]);  view_845 = None
        permute_741: "f32[4, 1024, 12, 513]" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:617 in forward, code: attn_probs = nn.functional.softmax(
        clone_148: "f32[4, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_741, memory_format = torch.contiguous_format);  permute_741 = None
        amax_11: "f32[4, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_148, [-1], True)
        sub_92: "f32[4, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_148, amax_11);  clone_148 = amax_11 = None
        exp_11: "f32[4, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
        sum_12: "f32[4, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_117: "f32[4, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:628 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_95: "f32[4, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_225, full_default_119, div_117);  unsqueeze_225 = full_default_119 = div_117 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:890 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        permute_743: "f32[4, 12, 1024, 513]" = torch.ops.aten.permute.default(where_95, [0, 2, 1, 3]);  where_95 = None
        clone_150: "f32[4, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_743, memory_format = torch.contiguous_format);  permute_743 = None
        view_848: "f32[48, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_150, [48, 4, 256, 513]);  clone_150 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_47: "f32[48, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_848, [0, 257], 0.0);  view_848 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:740 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_850: "f32[48, 4, 197120]" = torch.ops.aten.reshape.default(constant_pad_nd_47, [48, 4, -1]);  constant_pad_nd_47 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:743 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_2768: "f32[48, 4, 196864]" = torch.ops.aten.slice.Tensor(view_850, 2, 0, -256);  view_850 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_851: "f32[48, 4, 256, 769]" = torch.ops.aten.reshape.default(slice_2768, [48, 4, 256, 769]);  slice_2768 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_2772: "f32[48, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_851, 3, 0, -1);  view_851 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_226: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_2772, 4);  slice_2772 = None
        permute_745: "f32[48, 4, 256, 1, 768]" = torch.ops.aten.permute.default(unsqueeze_226, [0, 1, 2, 4, 3]);  unsqueeze_226 = None
        permute_747: "f32[48, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_745, [0, 1, 2, 4, 3]);  permute_745 = None
        view_852: "f32[192, 256, 768]" = torch.ops.aten.reshape.default(permute_747, [192, 256, 768]);  permute_747 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:549 in forward, code: value_vectors = self.value(hidden_states)
        clone_145: "f32[1024, 4, 768]" = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format);  permute_693 = None
        view_796: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_145, [4096, 768]);  clone_145 = None
        permute_696: "f32[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_46: "f32[4096, 768]" = torch.ops.aten.mm.default(view_796, permute_696);  view_796 = permute_696 = None
        view_797: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(mm_46, [1024, 4, 768]);  mm_46 = None
        add_167: "f32[1024, 4, 768]" = torch.ops.aten.add.Tensor(view_797, arg184_1);  view_797 = arg184_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        view_847: "f32[1024, 4, 12, 64]" = torch.ops.aten.reshape.default(add_167, [1024, 4, 12, 64]);  add_167 = None
        permute_742: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_847, [1, 0, 2, 3]);  view_847 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:898 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        permute_744: "f32[4, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_742, [0, 2, 1, 3]);  permute_742 = None
        view_849: "f32[48, 1024, 64]" = torch.ops.aten.reshape.default(permute_744, [48, 1024, 64]);  permute_744 = None
        
         # File: /data/users/sahanp/pytorch/torch/nn/functional.py:5096 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_46: "f32[48, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_849, [0, 0, 256, 256], -1.0);  view_849 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:912 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        as_strided_71: "f32[48, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_46, [48, 4, 768, 64], [98304, 16384, 64, 1]);  constant_pad_nd_46 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:916 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        unsqueeze_227: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_71, 4);  as_strided_71 = None
        permute_746: "f32[48, 4, 1, 64, 768]" = torch.ops.aten.permute.default(unsqueeze_227, [0, 1, 4, 3, 2]);  unsqueeze_227 = None
        permute_748: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_746, [0, 1, 4, 3, 2]);  permute_746 = None
        clone_151: "f32[48, 4, 768, 64, 1]" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
        view_853: "f32[192, 768, 64]" = torch.ops.aten.reshape.default(clone_151, [192, 768, 64]);  clone_151 = None
        bmm_23: "f32[192, 256, 64]" = torch.ops.aten.bmm.default(view_852, view_853);  view_852 = view_853 = None
        view_854: "f32[48, 4, 256, 1, 64]" = torch.ops.aten.reshape.default(bmm_23, [48, 4, 256, 1, 64]);  bmm_23 = None
        permute_749: "f32[48, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_854, [0, 1, 2, 4, 3]);  view_854 = None
        view_855: "f32[48, 4, 256, 64]" = torch.ops.aten.reshape.default(permute_749, [48, 4, 256, 64]);  permute_749 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:917 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        view_856: "f32[4, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_855, [4, 12, 1024, 64]);  view_855 = None
        permute_750: "f32[4, 1024, 12, 64]" = torch.ops.aten.permute.default(view_856, [0, 2, 1, 3]);  view_856 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:656 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        permute_751: "f32[1024, 4, 12, 64]" = torch.ops.aten.permute.default(permute_750, [1, 0, 2, 3]);  permute_750 = None
        clone_152: "f32[1024, 4, 12, 64]" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
        view_857: "f32[1024, 4, 768]" = torch.ops.aten.reshape.default(clone_152, [1024, 4, 768]);  clone_152 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:685 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_752: "f32[4, 1024, 768]" = torch.ops.aten.permute.default(view_857, [1, 0, 2]);  view_857 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1132 in forward, code: hidden_states = self.dense(hidden_states)
        clone_153: "f32[4, 1024, 768]" = torch.ops.aten.clone.default(permute_752, memory_format = torch.contiguous_format);  permute_752 = None
        view_858: "f32[4096, 768]" = torch.ops.aten.reshape.default(clone_153, [4096, 768]);  clone_153 = None
        permute_753: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_47: "f32[4096, 768]" = torch.ops.aten.mm.default(view_858, permute_753);  view_858 = permute_753 = None
        view_859: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(mm_47, [4, 1024, 768]);  mm_47 = None
        add_172: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_859, arg186_1);  view_859 = arg186_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1134 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_173: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(add_172, add_164);  add_172 = add_164 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_44: "f32[4, 1024, 1]" = var_mean_22[0]
        getitem_45: "f32[4, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_94: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_173, getitem_45);  add_173 = getitem_45 = None
        add_174: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_89: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_22);  sub_94 = rsqrt_22 = None
        mul_90: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_89, arg187_1);  mul_89 = arg187_1 = None
        add_175: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_90, arg188_1);  mul_90 = arg188_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_860: "f32[4096, 768]" = torch.ops.aten.reshape.default(add_175, [4096, 768])
        permute_754: "f32[768, 3072]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "f32[4096, 3072]" = torch.ops.aten.mm.default(view_860, permute_754);  view_860 = permute_754 = None
        add_tensor_1: "f32[4096, 3072]" = torch.ops.aten.add.Tensor(mm_default_1, arg190_1);  mm_default_1 = arg190_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1198 in forward, code: hidden_states = self.dense(hidden_states)
        view_861: "f32[4, 1024, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [4, 1024, 3072]);  add_tensor_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul_91: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_861, 0.5)
        mul_92: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_861, 0.7071067811865476);  view_861 = None
        erf_11: "f32[4, 1024, 3072]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_176: "f32[4, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_93: "f32[4, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_91, add_176);  mul_91 = add_176 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_862: "f32[4096, 3072]" = torch.ops.aten.reshape.default(mul_93, [4096, 3072]);  mul_93 = None
        permute_755: "f32[3072, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        
        # No stacktrace found for following nodes
        mm_default: "f32[4096, 768]" = torch.ops.aten.mm.default(view_862, permute_755);  view_862 = permute_755 = None
        add_tensor: "f32[4096, 768]" = torch.ops.aten.add.Tensor(mm_default, arg192_1);  mm_default = arg192_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1212 in forward, code: hidden_states = self.dense(hidden_states)
        view_863: "f32[4, 1024, 768]" = torch.ops.aten.reshape.default(add_tensor, [4, 1024, 768]);  add_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1214 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_177: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(view_863, add_175);  view_863 = add_175 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_46: "f32[4, 1024, 1]" = var_mean_23[0]
        getitem_47: "f32[4, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_95: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(add_177, getitem_47);  add_177 = getitem_47 = None
        add_178: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_94: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_23);  sub_95 = rsqrt_23 = None
        mul_95: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_94, arg193_1);  mul_94 = arg193_1 = None
        add_179: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_95, arg194_1);  mul_95 = arg194_1 = None
        return (add_179,)
        