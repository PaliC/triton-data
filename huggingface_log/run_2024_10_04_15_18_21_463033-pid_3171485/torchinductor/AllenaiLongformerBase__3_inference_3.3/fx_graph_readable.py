class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 1024, 768]", arg1_1: "f32[768, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[50265, 768]", arg6_1: "f32[50265]", arg7_1: "i64[4, 1024]"):
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1383 in forward, code: x = self.dense(features)
        view: "f32[4096, 768]" = torch.ops.aten.view.default(arg0_1, [4096, 768]);  arg0_1 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "f32[4096, 768]" = torch.ops.aten.addmm.default(arg2_1, view, permute);  arg2_1 = view = permute = None
        view_1: "f32[4, 1024, 768]" = torch.ops.aten.view.default(addmm, [4, 1024, 768]);  addmm = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/activations.py:78 in forward, code: return self.act(input)
        mul: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        mul_1: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476);  view_1 = None
        erf: "f32[4, 1024, 768]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1385 in forward, code: x = self.layer_norm(x)
        var_mean = torch.ops.aten.var_mean.correction(mul_2, [2], correction = 0, keepdim = True)
        getitem: "f32[4, 1024, 1]" = var_mean[0]
        getitem_1: "f32[4, 1024, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[4, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[4, 1024, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[4, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_2, getitem_1);  mul_2 = getitem_1 = None
        mul_3: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_4: "f32[4, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg3_1);  mul_3 = arg3_1 = None
        add_2: "f32[4, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, arg4_1);  mul_4 = arg4_1 = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1388 in forward, code: x = self.decoder(x)
        view_2: "f32[4096, 768]" = torch.ops.aten.view.default(add_2, [4096, 768]);  add_2 = None
        permute_1: "f32[768, 50265]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        
        # No stacktrace found for following nodes
        full_default_2: "f32[768, 3]" = torch.ops.aten.full.default([768, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f32[768, 50268]" = torch.ops.aten.cat.default([permute_1, full_default_2], 1);  permute_1 = full_default_2 = None
        full_default_3: "f32[3]" = torch.ops.aten.full.default([3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default_1: "f32[50268]" = torch.ops.aten.cat.default([arg6_1, full_default_3]);  arg6_1 = full_default_3 = None
        addmm_default: "f32[4096, 50268]" = torch.ops.aten.addmm.default(cat_default_1, view_2, cat_default);  cat_default_1 = view_2 = cat_default = None
        slice_tensor: "f32[4096, 50265]" = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -3);  addmm_default = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1388 in forward, code: x = self.decoder(x)
        view_3: "f32[4, 1024, 50265]" = torch.ops.aten.view.default(slice_tensor, [4, 1024, 50265]);  slice_tensor = None
        
         # File: /home/sahanp/.conda/envs/pytorch-3.10/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1846 in torch_dynamo_resume_in_forward_at_1826, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_4: "f32[4096, 50265]" = torch.ops.aten.view.default(view_3, [-1, 50265])
        view_5: "i64[4096]" = torch.ops.aten.view.default(arg7_1, [-1]);  arg7_1 = None
        amax: "f32[4096, 1]" = torch.ops.aten.amax.default(view_4, [1], True)
        sub_1: "f32[4096, 50265]" = torch.ops.aten.sub.Tensor(view_4, amax);  view_4 = amax = None
        exp: "f32[4096, 50265]" = torch.ops.aten.exp.default(sub_1)
        sum_1: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_2: "f32[4096, 50265]" = torch.ops.aten.sub.Tensor(sub_1, log);  sub_1 = log = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_5, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096]" = torch.ops.aten.where.self(ne, view_5, full_default);  ne = full_default = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_2, 1, unsqueeze);  sub_2 = unsqueeze = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        ne_1: "b8[4096]" = torch.ops.aten.ne.Scalar(view_5, -100)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[4096]" = torch.ops.aten.where.self(ne_1, neg, full_default_1);  ne_1 = neg = full_default_1 = None
        ne_2: "b8[4096]" = torch.ops.aten.ne.Scalar(view_5, -100);  view_5 = None
        sum_2: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_3, convert_element_type);  sum_3 = convert_element_type = None
        return (div, view_3)
        