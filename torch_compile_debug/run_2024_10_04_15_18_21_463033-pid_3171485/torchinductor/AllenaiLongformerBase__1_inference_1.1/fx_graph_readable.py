class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 1024]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: is_index_masked = attention_mask < 0
        lt: "b8[4, 1024]" = torch.ops.aten.lt.Scalar(arg0_1, 0)
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1279 in forward, code: is_index_global_attn = attention_mask > 0
        gt: "b8[4, 1024]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1282 in forward, code: is_global_attn = is_index_global_attn.flatten().any().item()
        view: "b8[4096]" = torch.ops.aten.view.default(gt, [4096])
        any_1: "b8[]" = torch.ops.aten.any.default(view);  view = None
        return (any_1, lt, gt)
        