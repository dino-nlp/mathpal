class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "bf16[262400, 2048]", primals_2: "Sym(s0)", primals_3: "i64[1, s0]", primals_4: "Sym(s1)", primals_5: "i64[1, s1]", primals_6: "Sym(s3)", primals_7: "f32[1, s3, 2048]", primals_8: "i64[]"):
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:517 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels = torch.empty_like(output_labels, device = device)
        empty: "i64[1, s0]" = torch.ops.aten.empty.memory_format([1, primals_2], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute: "i64[1, s0]" = torch.ops.aten.permute.default(empty, [0, 1]);  empty = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:518 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., :-1] = output_labels[..., 1:]
        slice_1: "i64[1, s0 - 1]" = torch.ops.aten.slice.Tensor(primals_3, 1, 1, 9223372036854775807);  primals_3 = None
        
        # No stacktrace found for following nodes
        slice_scatter_default: "i64[1, s0]" = torch.ops.aten.slice_scatter.default(permute, slice_1, 1, 0, -1);  permute = slice_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:521 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., :-1][mask[..., 1:] == 0] = -100
        slice_5: "i64[1, s1 - 1]" = torch.ops.aten.slice.Tensor(primals_5, 1, 1, 9223372036854775807);  primals_5 = None
        eq_14: "b8[1, s1 - 1]" = torch.ops.aten.eq.Scalar(slice_5, 0);  slice_5 = None
        full_default: "i64[]" = torch.ops.aten.full.default([], -100, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        slice_6: "i64[1, s0 - 1]" = torch.ops.aten.slice.Tensor(slice_scatter_default, 1, 0, -1)
        index_put: "i64[1, s0 - 1]" = torch.ops.aten.index_put_.default(slice_6, [eq_14], full_default);  slice_6 = eq_14 = None
        
        # No stacktrace found for following nodes
        slice_scatter_default_1: "i64[1, s0]" = torch.ops.aten.slice_scatter.default(slice_scatter_default, index_put, 1, 0, -1);  slice_scatter_default = index_put = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:523 in _unsloth_compiled_fused_ce_loss_function, code: shift_labels[..., -1] = -100
        select_1: "i64[1]" = torch.ops.aten.select.int(slice_scatter_default_1, 1, -1)
        copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
        
        # No stacktrace found for following nodes
        select_scatter_default: "i64[1, s0]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, copy_1, 1, -1);  slice_scatter_default_1 = copy_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:528 in _unsloth_compiled_fused_ce_loss_function, code: hidden_states = hidden_states.view(-1, hd)
        view_1: "f32[s3, 2048]" = torch.ops.aten.reshape.default(primals_7, [-1, 2048]);  primals_7 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        add_31: "Sym(s3 + 9)" = primals_6 + 9
        sub_14: "Sym(s3 + 8)" = add_31 - 1;  add_31 = None
        floordiv: "Sym(((s3 + 8)//9))" = sub_14 // 9;  sub_14 = None
        split = torch.ops.aten.split.Tensor(view_1, floordiv);  view_1 = floordiv = None
        getitem: "f32[((s3 + 8)//9), 2048]" = split[0]
        getitem_1: "f32[((s3 + 8)//9), 2048]" = split[1]
        getitem_2: "f32[((s3 + 8)//9), 2048]" = split[2]
        getitem_3: "f32[((s3 + 8)//9), 2048]" = split[3]
        getitem_4: "f32[((s3 + 8)//9), 2048]" = split[4]
        getitem_5: "f32[((s3 + 8)//9), 2048]" = split[5]
        getitem_6: "f32[((s3 + 8)//9), 2048]" = split[6]
        getitem_7: "f32[((s3 + 8)//9), 2048]" = split[7]
        getitem_8: "f32[s3 - 8*(((s3 + 8)//9)), 2048]" = split[8];  split = None
        sym_size_int_8: "Sym(s3 - 8*(((s3 + 8)//9)))" = torch.ops.aten.sym_size.int(getitem_8, 0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:531 in _unsloth_compiled_fused_ce_loss_function, code: __shift_labels = torch.chunk(shift_labels,  n_chunks, dim = 0)
        add_59: "Sym(s0 + 9)" = primals_2 + 9
        sub_24: "Sym(s0 + 8)" = add_59 - 1;  add_59 = None
        floordiv_1: "Sym(((s0 + 8)//9))" = sub_24 // 9;  sub_24 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type: "bf16[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(getitem, torch.bfloat16);  getitem = None
        permute_1: "bf16[2048, 262400]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mm.default(convert_element_type, permute_1);  convert_element_type = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div);  div = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_8: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_tensor_16: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1);  convert_element_type_default_8 = None
        amax_default_8: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.amax.default(mul_tensor_16, [1], True)
        sub_tensor_8: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_16, amax_default_8);  mul_tensor_16 = None
        mul_tensor_17: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_8, 30.0);  sub_tensor_8 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(mul_tensor_17)
        sum_1: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_42: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_17, log);  mul_tensor_17 = None
        view_2: "i64[s0]" = torch.ops.aten.reshape.default(select_scatter_default, [-1]);  select_scatter_default = None
        split_2 = torch.ops.aten.split.Tensor(view_2, floordiv_1);  view_2 = floordiv_1 = None
        getitem_18: "i64[((s0 + 8)//9)]" = split_2[0]
        ne_4: "b8[((s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_18, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_4, getitem_18, full_default_2)
        unsqueeze: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_42, 1, unsqueeze);  sub_42 = unsqueeze = None
        squeeze: "f32[((s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[((s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_4, neg, full_default_3);  ne_4 = neg = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_105: "f32[]" = torch.ops.aten.add.Tensor(sum_3, 0.0);  sum_3 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type_5: "bf16[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.bfloat16);  getitem_1 = None
        mm_1: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mm.default(convert_element_type_5, permute_1);  convert_element_type_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_1: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_1, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_1: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_1);  div_1 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_7: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_1, torch.float32);  tanh_1 = None
        mul_tensor_14: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1);  convert_element_type_default_7 = None
        amax_default_7: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.amax.default(mul_tensor_14, [1], True)
        sub_tensor_7: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = None
        mul_tensor_15: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_7, 30.0);  sub_tensor_7 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp_1: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(mul_tensor_15)
        sum_4: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_53: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_15, log_1);  mul_tensor_15 = None
        getitem_28: "i64[((s0 + 8)//9)]" = split_2[1]
        ne_10: "b8[((s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_28, -100)
        where_2: "i64[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_10, getitem_28, full_default_2)
        unsqueeze_1: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_53, 1, unsqueeze_1);  sub_53 = unsqueeze_1 = None
        squeeze_1: "f32[((s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1: "f32[((s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        where_3: "f32[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_10, neg_1, full_default_3);  ne_10 = neg_1 = None
        sum_6: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_133: "f32[]" = torch.ops.aten.add.Tensor(add_105, sum_6);  add_105 = sum_6 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type_10: "bf16[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(getitem_2, torch.bfloat16);  getitem_2 = None
        mm_2: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mm.default(convert_element_type_10, permute_1);  convert_element_type_10 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_2: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_2, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_2: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_2);  div_2 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_6: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_2, torch.float32);  tanh_2 = None
        mul_tensor_12: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1);  convert_element_type_default_6 = None
        amax_default_6: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.amax.default(mul_tensor_12, [1], True)
        sub_tensor_6: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = None
        mul_tensor_13: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_6, 30.0);  sub_tensor_6 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp_2: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(mul_tensor_13)
        sum_7: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True);  exp_2 = None
        log_2: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_64: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_13, log_2);  mul_tensor_13 = None
        getitem_38: "i64[((s0 + 8)//9)]" = split_2[2]
        ne_16: "b8[((s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_38, -100)
        where_4: "i64[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_16, getitem_38, full_default_2)
        unsqueeze_2: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_4, 1);  where_4 = None
        gather_2: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_64, 1, unsqueeze_2);  sub_64 = unsqueeze_2 = None
        squeeze_2: "f32[((s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_2, 1);  gather_2 = None
        neg_2: "f32[((s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        where_5: "f32[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_16, neg_2, full_default_3);  ne_16 = neg_2 = None
        sum_9: "f32[]" = torch.ops.aten.sum.default(where_5);  where_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_161: "f32[]" = torch.ops.aten.add.Tensor(add_133, sum_9);  add_133 = sum_9 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type_15: "bf16[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.bfloat16);  getitem_3 = None
        mm_3: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mm.default(convert_element_type_15, permute_1);  convert_element_type_15 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_3: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_3, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_3: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_3);  div_3 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_5: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_3, torch.float32);  tanh_3 = None
        mul_tensor_10: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1);  convert_element_type_default_5 = None
        amax_default_5: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.amax.default(mul_tensor_10, [1], True)
        sub_tensor_5: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = None
        mul_tensor_11: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_5, 30.0);  sub_tensor_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp_3: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(mul_tensor_11)
        sum_10: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [1], True);  exp_3 = None
        log_3: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.log.default(sum_10);  sum_10 = None
        sub_75: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_11, log_3);  mul_tensor_11 = None
        getitem_48: "i64[((s0 + 8)//9)]" = split_2[3]
        ne_22: "b8[((s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_48, -100)
        where_6: "i64[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_22, getitem_48, full_default_2)
        unsqueeze_3: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather_3: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_75, 1, unsqueeze_3);  sub_75 = unsqueeze_3 = None
        squeeze_3: "f32[((s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_3, 1);  gather_3 = None
        neg_3: "f32[((s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        where_7: "f32[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_22, neg_3, full_default_3);  ne_22 = neg_3 = None
        sum_12: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_189: "f32[]" = torch.ops.aten.add.Tensor(add_161, sum_12);  add_161 = sum_12 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type_20: "bf16[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(getitem_4, torch.bfloat16);  getitem_4 = None
        mm_4: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mm.default(convert_element_type_20, permute_1);  convert_element_type_20 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_4: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_4, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_4: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_4);  div_4 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_4: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_4, torch.float32);  tanh_4 = None
        mul_tensor_8: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1);  convert_element_type_default_4 = None
        amax_default_4: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.amax.default(mul_tensor_8, [1], True)
        sub_tensor_4: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = None
        mul_tensor_9: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_4, 30.0);  sub_tensor_4 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp_4: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(mul_tensor_9)
        sum_13: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True);  exp_4 = None
        log_4: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_86: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_9, log_4);  mul_tensor_9 = None
        getitem_58: "i64[((s0 + 8)//9)]" = split_2[4]
        ne_28: "b8[((s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_58, -100)
        where_8: "i64[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_28, getitem_58, full_default_2)
        unsqueeze_4: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
        gather_4: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_86, 1, unsqueeze_4);  sub_86 = unsqueeze_4 = None
        squeeze_4: "f32[((s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_4, 1);  gather_4 = None
        neg_4: "f32[((s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
        where_9: "f32[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_28, neg_4, full_default_3);  ne_28 = neg_4 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_9);  where_9 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_217: "f32[]" = torch.ops.aten.add.Tensor(add_189, sum_15);  add_189 = sum_15 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type_25: "bf16[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.bfloat16);  getitem_5 = None
        mm_5: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mm.default(convert_element_type_25, permute_1);  convert_element_type_25 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_5: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_5, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_5: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_5);  div_5 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_3: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_5, torch.float32);  tanh_5 = None
        mul_tensor_6: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        amax_default_3: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.amax.default(mul_tensor_6, [1], True)
        sub_tensor_3: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = None
        mul_tensor_7: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_3, 30.0);  sub_tensor_3 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp_5: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(mul_tensor_7)
        sum_16: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [1], True);  exp_5 = None
        log_5: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_97: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_7, log_5);  mul_tensor_7 = None
        getitem_68: "i64[((s0 + 8)//9)]" = split_2[5]
        ne_34: "b8[((s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_68, -100)
        where_10: "i64[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_34, getitem_68, full_default_2)
        unsqueeze_5: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_10, 1);  where_10 = None
        gather_5: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_97, 1, unsqueeze_5);  sub_97 = unsqueeze_5 = None
        squeeze_5: "f32[((s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_5, 1);  gather_5 = None
        neg_5: "f32[((s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_5);  squeeze_5 = None
        where_11: "f32[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_34, neg_5, full_default_3);  ne_34 = neg_5 = None
        sum_18: "f32[]" = torch.ops.aten.sum.default(where_11);  where_11 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_245: "f32[]" = torch.ops.aten.add.Tensor(add_217, sum_18);  add_217 = sum_18 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type_30: "bf16[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(getitem_6, torch.bfloat16);  getitem_6 = None
        mm_6: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mm.default(convert_element_type_30, permute_1);  convert_element_type_30 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_6: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_6, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_6: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_6);  div_6 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_2: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_6, torch.float32);  tanh_6 = None
        mul_tensor_4: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        amax_default_2: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.amax.default(mul_tensor_4, [1], True)
        sub_tensor_2: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = None
        mul_tensor_5: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_2, 30.0);  sub_tensor_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp_6: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(mul_tensor_5)
        sum_19: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log_6: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_108: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_5, log_6);  mul_tensor_5 = None
        getitem_78: "i64[((s0 + 8)//9)]" = split_2[6]
        ne_40: "b8[((s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_78, -100)
        where_12: "i64[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_40, getitem_78, full_default_2)
        unsqueeze_6: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_6: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_108, 1, unsqueeze_6);  sub_108 = unsqueeze_6 = None
        squeeze_6: "f32[((s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_6, 1);  gather_6 = None
        neg_6: "f32[((s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_6);  squeeze_6 = None
        where_13: "f32[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_40, neg_6, full_default_3);  ne_40 = neg_6 = None
        sum_21: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_273: "f32[]" = torch.ops.aten.add.Tensor(add_245, sum_21);  add_245 = sum_21 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type_35: "bf16[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.bfloat16);  getitem_7 = None
        mm_7: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mm.default(convert_element_type_35, permute_1);  convert_element_type_35 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_7: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_7, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_7: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_7);  div_7 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_1: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_7, torch.float32);  tanh_7 = None
        mul_tensor_2: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default_1: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.amax.default(mul_tensor_2, [1], True)
        sub_tensor_1: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = None
        mul_tensor_3: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_1, 30.0);  sub_tensor_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp_7: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(mul_tensor_3)
        sum_22: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [1], True);  exp_7 = None
        log_7: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.log.default(sum_22);  sum_22 = None
        sub_119: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_3, log_7);  mul_tensor_3 = None
        getitem_88: "i64[((s0 + 8)//9)]" = split_2[7]
        ne_46: "b8[((s0 + 8)//9)]" = torch.ops.aten.ne.Scalar(getitem_88, -100)
        where_14: "i64[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_46, getitem_88, full_default_2)
        unsqueeze_7: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(where_14, 1);  where_14 = None
        gather_7: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.gather.default(sub_119, 1, unsqueeze_7);  sub_119 = unsqueeze_7 = None
        squeeze_7: "f32[((s0 + 8)//9)]" = torch.ops.aten.squeeze.dim(gather_7, 1);  gather_7 = None
        neg_7: "f32[((s0 + 8)//9)]" = torch.ops.aten.neg.default(squeeze_7);  squeeze_7 = None
        where_15: "f32[((s0 + 8)//9)]" = torch.ops.aten.where.self(ne_46, neg_7, full_default_3);  ne_46 = neg_7 = None
        sum_24: "f32[]" = torch.ops.aten.sum.default(where_15);  where_15 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_301: "f32[]" = torch.ops.aten.add.Tensor(add_273, sum_24);  add_273 = sum_24 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        convert_element_type_40: "bf16[s3 - 8*(((s3 + 8)//9)), 2048]" = torch.ops.prims.convert_element_type.default(getitem_8, torch.bfloat16);  getitem_8 = None
        mm_8: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mm.default(convert_element_type_40, permute_1);  convert_element_type_40 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_8: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.div.Tensor(mm_8, 30.0)
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_8: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.tanh.default(div_8);  div_8 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.prims.convert_element_type.default(tanh_8, torch.float32);  tanh_8 = None
        mul_tensor: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1);  convert_element_type_default = None
        amax_default: "f32[s3 - 8*(((s3 + 8)//9)), 1]" = torch.ops.aten.amax.default(mul_tensor, [1], True)
        sub_tensor: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = None
        mul_tensor_1: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor, 30.0);  sub_tensor = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        exp_8: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.exp.default(mul_tensor_1)
        sum_25: "f32[s3 - 8*(((s3 + 8)//9)), 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True);  exp_8 = None
        log_8: "f32[s3 - 8*(((s3 + 8)//9)), 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_130: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_1, log_8);  mul_tensor_1 = None
        getitem_98: "i64[s0 - 8*(((s0 + 8)//9))]" = split_2[8];  split_2 = None
        ne_52: "b8[s0 - 8*(((s0 + 8)//9))]" = torch.ops.aten.ne.Scalar(getitem_98, -100)
        where_16: "i64[s0 - 8*(((s0 + 8)//9))]" = torch.ops.aten.where.self(ne_52, getitem_98, full_default_2)
        unsqueeze_8: "i64[s0 - 8*(((s0 + 8)//9)), 1]" = torch.ops.aten.unsqueeze.default(where_16, 1);  where_16 = None
        gather_8: "f32[s0 - 8*(((s0 + 8)//9)), 1]" = torch.ops.aten.gather.default(sub_130, 1, unsqueeze_8);  sub_130 = unsqueeze_8 = None
        squeeze_8: "f32[s0 - 8*(((s0 + 8)//9))]" = torch.ops.aten.squeeze.dim(gather_8, 1);  gather_8 = None
        neg_8: "f32[s0 - 8*(((s0 + 8)//9))]" = torch.ops.aten.neg.default(squeeze_8);  squeeze_8 = None
        where_17: "f32[s0 - 8*(((s0 + 8)//9))]" = torch.ops.aten.where.self(ne_52, neg_8, full_default_3);  ne_52 = neg_8 = full_default_3 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_17);  where_17 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:553 in _unsloth_compiled_fused_ce_loss_function, code: loss += torch.nn.functional.cross_entropy(
        add_329: "f32[]" = torch.ops.aten.add.Tensor(add_301, sum_27);  add_301 = sum_27 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:560 in _unsloth_compiled_fused_ce_loss_function, code: loss = loss / torch.tensor(divisor, dtype = torch.float32)
        convert_element_type_45: "f32[]" = torch.ops.prims.convert_element_type.default(primals_8, torch.float32);  primals_8 = None
        div_9: "f32[]" = torch.ops.aten.div.Tensor(add_329, convert_element_type_45);  add_329 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        unsqueeze_9: "i64[s0 - 8*(((s0 + 8)//9)), 1]" = torch.ops.aten.unsqueeze.default(getitem_98, 1);  getitem_98 = None
        ne_55: "b8[s0 - 8*(((s0 + 8)//9)), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_9, -100)
        where_18: "i64[s0 - 8*(((s0 + 8)//9)), 1]" = torch.ops.aten.where.self(ne_55, unsqueeze_9, full_default_2);  unsqueeze_9 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        permute_10: "bf16[262400, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        unsqueeze_10: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_88, 1);  getitem_88 = None
        ne_57: "b8[((s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_10, -100)
        where_20: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_57, unsqueeze_10, full_default_2);  unsqueeze_10 = None
        unsqueeze_11: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_78, 1);  getitem_78 = None
        ne_59: "b8[((s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_22: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_59, unsqueeze_11, full_default_2);  unsqueeze_11 = None
        unsqueeze_12: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_68, 1);  getitem_68 = None
        ne_61: "b8[((s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_12, -100)
        where_24: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_61, unsqueeze_12, full_default_2);  unsqueeze_12 = None
        unsqueeze_13: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_58, 1);  getitem_58 = None
        ne_63: "b8[((s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_13, -100)
        where_26: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_63, unsqueeze_13, full_default_2);  unsqueeze_13 = None
        unsqueeze_14: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_48, 1);  getitem_48 = None
        ne_65: "b8[((s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_14, -100)
        where_28: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_65, unsqueeze_14, full_default_2);  unsqueeze_14 = None
        unsqueeze_15: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_38, 1);  getitem_38 = None
        ne_67: "b8[((s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_15, -100)
        where_30: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_67, unsqueeze_15, full_default_2);  unsqueeze_15 = None
        unsqueeze_16: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_28, 1);  getitem_28 = None
        ne_69: "b8[((s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_16, -100)
        where_32: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_69, unsqueeze_16, full_default_2);  unsqueeze_16 = None
        unsqueeze_17: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.unsqueeze.default(getitem_18, 1);  getitem_18 = None
        ne_71: "b8[((s0 + 8)//9), 1]" = torch.ops.aten.ne.Scalar(unsqueeze_17, -100)
        where_34: "i64[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_71, unsqueeze_17, full_default_2);  unsqueeze_17 = full_default_2 = None
        return (div_9, mm, amax_default_8, log, mm_1, amax_default_7, log_1, mm_2, amax_default_6, log_2, mm_3, amax_default_5, log_3, mm_4, amax_default_4, log_4, mm_5, amax_default_3, log_5, mm_6, amax_default_2, log_6, mm_7, amax_default_1, log_7, mm_8, amax_default, log_8, convert_element_type_45, ne_55, where_18, permute_10, ne_57, where_20, ne_59, where_22, ne_61, where_24, ne_63, where_26, ne_65, where_28, ne_67, where_30, ne_69, where_32, ne_71, where_34, primals_6, primals_2, sym_size_int_8)
        