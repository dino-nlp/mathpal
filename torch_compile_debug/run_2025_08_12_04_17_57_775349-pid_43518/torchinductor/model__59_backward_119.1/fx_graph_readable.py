class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "Sym(s3)", primals_2: "Sym(s0)", sym_size_int_8: "Sym(s3 - 8*(((s3 + 8)//9)))", mm: "bf16[((s3 + 8)//9), 262400]", amax_default_8: "f32[((s3 + 8)//9), 1]", log: "f32[((s3 + 8)//9), 1]", mm_1: "bf16[((s3 + 8)//9), 262400]", amax_default_7: "f32[((s3 + 8)//9), 1]", log_1: "f32[((s3 + 8)//9), 1]", mm_2: "bf16[((s3 + 8)//9), 262400]", amax_default_6: "f32[((s3 + 8)//9), 1]", log_2: "f32[((s3 + 8)//9), 1]", mm_3: "bf16[((s3 + 8)//9), 262400]", amax_default_5: "f32[((s3 + 8)//9), 1]", log_3: "f32[((s3 + 8)//9), 1]", mm_4: "bf16[((s3 + 8)//9), 262400]", amax_default_4: "f32[((s3 + 8)//9), 1]", log_4: "f32[((s3 + 8)//9), 1]", mm_5: "bf16[((s3 + 8)//9), 262400]", amax_default_3: "f32[((s3 + 8)//9), 1]", log_5: "f32[((s3 + 8)//9), 1]", mm_6: "bf16[((s3 + 8)//9), 262400]", amax_default_2: "f32[((s3 + 8)//9), 1]", log_6: "f32[((s3 + 8)//9), 1]", mm_7: "bf16[((s3 + 8)//9), 262400]", amax_default_1: "f32[((s3 + 8)//9), 1]", log_7: "f32[((s3 + 8)//9), 1]", mm_8: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]", amax_default: "f32[s3 - 8*(((s3 + 8)//9)), 1]", log_8: "f32[s3 - 8*(((s3 + 8)//9)), 1]", convert_element_type_45: "f32[]", ne_55: "b8[s0 - 8*(((s0 + 8)//9)), 1]", where_18: "i64[s0 - 8*(((s0 + 8)//9)), 1]", permute_10: "bf16[262400, 2048]", ne_57: "b8[((s0 + 8)//9), 1]", where_20: "i64[((s0 + 8)//9), 1]", ne_59: "b8[((s0 + 8)//9), 1]", where_22: "i64[((s0 + 8)//9), 1]", ne_61: "b8[((s0 + 8)//9), 1]", where_24: "i64[((s0 + 8)//9), 1]", ne_63: "b8[((s0 + 8)//9), 1]", where_26: "i64[((s0 + 8)//9), 1]", ne_65: "b8[((s0 + 8)//9), 1]", where_28: "i64[((s0 + 8)//9), 1]", ne_67: "b8[((s0 + 8)//9), 1]", where_30: "i64[((s0 + 8)//9), 1]", ne_69: "b8[((s0 + 8)//9), 1]", where_32: "i64[((s0 + 8)//9), 1]", ne_71: "b8[((s0 + 8)//9), 1]", where_34: "i64[((s0 + 8)//9), 1]", tangents_1: "f32[]"):
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:560 in _unsloth_compiled_fused_ce_loss_function, code: loss = loss / torch.tensor(divisor, dtype = torch.float32)
        div_10: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_45);  tangents_1 = convert_element_type_45 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        full: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.full.default([sym_size_int_8, 262400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  sym_size_int_8 = None
        scatter: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.scatter.value(full, 1, where_18, -1.0);  full = where_18 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f32[s0 - 8*(((s0 + 8)//9)), 1]" = torch.ops.aten.where.self(ne_55, div_10, full_default_3);  ne_55 = None
        mul_246: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(scatter, where_19);  scatter = where_19 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_8: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.div.Tensor(mm_8, 30.0);  mm_8 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_8: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.tanh.default(div_8);  div_8 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.prims.convert_element_type.default(tanh_8, torch.float32);  tanh_8 = None
        mul_tensor: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1)
        sub_tensor: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor, 30.0);  sub_tensor = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_130: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_1, log_8);  mul_tensor_1 = log_8 = None
        exp_9: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.exp.default(sub_130);  sub_130 = None
        sum_28: "f32[s3 - 8*(((s3 + 8)//9)), 1]" = torch.ops.aten.sum.dim_IntList(mul_246, [1], True)
        mul_247: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(exp_9, sum_28);  exp_9 = sum_28 = None
        sub_133: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.sub.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_46: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.prims.convert_element_type.default(sub_133, torch.bfloat16);  sub_133 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_248: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 30.0);  convert_element_type_46 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_47: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.prims.convert_element_type.default(mul_248, torch.float32);  mul_248 = None
        mul_249: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default);  convert_element_type_default = None
        sub_134: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.sub.Tensor(1, mul_249);  mul_249 = None
        mul_250: "f32[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_47, sub_134);  convert_element_type_47 = sub_134 = None
        convert_element_type_49: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_11: "bf16[s3 - 8*(((s3 + 8)//9)), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_49, 30.0);  convert_element_type_49 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_9: "bf16[s3 - 8*(((s3 + 8)//9)), 2048]" = torch.ops.aten.mm.default(div_11, permute_10);  div_11 = None
        convert_element_type_52: "f32[s3 - 8*(((s3 + 8)//9)), 2048]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        add_31: "Sym(s3 + 9)" = primals_6 + 9
        sub_14: "Sym(s3 + 8)" = add_31 - 1;  add_31 = None
        floordiv: "Sym(((s3 + 8)//9))" = sub_14 // 9;  sub_14 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        full_default_23: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.full.default([floordiv, 262400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  floordiv = None
        scatter_1: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.scatter.value(full_default_23, 1, where_20, -1.0);  where_20 = None
        where_21: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_57, div_10, full_default_3);  ne_57 = None
        mul_251: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(scatter_1, where_21);  scatter_1 = where_21 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_7: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_7, 30.0);  mm_7 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_7: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_7);  div_7 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_1: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_7, torch.float32);  tanh_7 = None
        mul_tensor_2: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1)
        sub_tensor_1: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_1, 30.0);  sub_tensor_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_119: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_3, log_7);  mul_tensor_3 = log_7 = None
        exp_10: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(sub_119);  sub_119 = None
        sum_29: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [1], True)
        mul_252: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(exp_10, sum_29);  exp_10 = sum_29 = None
        sub_135: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_53: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(sub_135, torch.bfloat16);  sub_135 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_253: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 30.0);  convert_element_type_53 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_54: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_253, torch.float32);  mul_253 = None
        mul_254: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1);  convert_element_type_default_1 = None
        sub_136: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(1, mul_254);  mul_254 = None
        mul_255: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_54, sub_136);  convert_element_type_54 = sub_136 = None
        convert_element_type_56: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_255, torch.bfloat16);  mul_255 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_12: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_56, 30.0);  convert_element_type_56 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_10: "bf16[((s3 + 8)//9), 2048]" = torch.ops.aten.mm.default(div_12, permute_10);  div_12 = None
        convert_element_type_59: "f32[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_2: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.scatter.value(full_default_23, 1, where_22, -1.0);  where_22 = None
        where_23: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_59, div_10, full_default_3);  ne_59 = None
        mul_256: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(scatter_2, where_23);  scatter_2 = where_23 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_6: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_6, 30.0);  mm_6 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_6: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_6);  div_6 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_2: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_6, torch.float32);  tanh_6 = None
        mul_tensor_4: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1)
        sub_tensor_2: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_2, 30.0);  sub_tensor_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_108: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_5, log_6);  mul_tensor_5 = log_6 = None
        exp_11: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(sub_108);  sub_108 = None
        sum_30: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [1], True)
        mul_257: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(exp_11, sum_30);  exp_11 = sum_30 = None
        sub_137: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_60: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(sub_137, torch.bfloat16);  sub_137 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_258: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 30.0);  convert_element_type_60 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_61: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_258, torch.float32);  mul_258 = None
        mul_259: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default_2);  convert_element_type_default_2 = None
        sub_138: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(1, mul_259);  mul_259 = None
        mul_260: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_61, sub_138);  convert_element_type_61 = sub_138 = None
        convert_element_type_63: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_260, torch.bfloat16);  mul_260 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_13: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_63, 30.0);  convert_element_type_63 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_11: "bf16[((s3 + 8)//9), 2048]" = torch.ops.aten.mm.default(div_13, permute_10);  div_13 = None
        convert_element_type_66: "f32[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_3: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.scatter.value(full_default_23, 1, where_24, -1.0);  where_24 = None
        where_25: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_61, div_10, full_default_3);  ne_61 = None
        mul_261: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(scatter_3, where_25);  scatter_3 = where_25 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_5: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_5, 30.0);  mm_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_5: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_5);  div_5 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_3: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_5, torch.float32);  tanh_5 = None
        mul_tensor_6: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1)
        sub_tensor_3: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_3, 30.0);  sub_tensor_3 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_97: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_7, log_5);  mul_tensor_7 = log_5 = None
        exp_12: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(sub_97);  sub_97 = None
        sum_31: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [1], True)
        mul_262: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(exp_12, sum_31);  exp_12 = sum_31 = None
        sub_139: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_67: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(sub_139, torch.bfloat16);  sub_139 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_263: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 30.0);  convert_element_type_67 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_68: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_263, torch.float32);  mul_263 = None
        mul_264: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, convert_element_type_default_3);  convert_element_type_default_3 = None
        sub_140: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(1, mul_264);  mul_264 = None
        mul_265: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_68, sub_140);  convert_element_type_68 = sub_140 = None
        convert_element_type_70: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_265, torch.bfloat16);  mul_265 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_14: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_70, 30.0);  convert_element_type_70 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_12: "bf16[((s3 + 8)//9), 2048]" = torch.ops.aten.mm.default(div_14, permute_10);  div_14 = None
        convert_element_type_73: "f32[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_4: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.scatter.value(full_default_23, 1, where_26, -1.0);  where_26 = None
        where_27: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_63, div_10, full_default_3);  ne_63 = None
        mul_266: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(scatter_4, where_27);  scatter_4 = where_27 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_4: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_4, 30.0);  mm_4 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_4: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_4);  div_4 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_4: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_4, torch.float32);  tanh_4 = None
        mul_tensor_8: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1)
        sub_tensor_4: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_4, 30.0);  sub_tensor_4 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_86: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_9, log_4);  mul_tensor_9 = log_4 = None
        exp_13: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(sub_86);  sub_86 = None
        sum_32: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [1], True)
        mul_267: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(exp_13, sum_32);  exp_13 = sum_32 = None
        sub_141: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_74: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(sub_141, torch.bfloat16);  sub_141 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_268: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 30.0);  convert_element_type_74 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_75: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_268, torch.float32);  mul_268 = None
        mul_269: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, convert_element_type_default_4);  convert_element_type_default_4 = None
        sub_142: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(1, mul_269);  mul_269 = None
        mul_270: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_75, sub_142);  convert_element_type_75 = sub_142 = None
        convert_element_type_77: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_270, torch.bfloat16);  mul_270 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_15: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_77, 30.0);  convert_element_type_77 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_13: "bf16[((s3 + 8)//9), 2048]" = torch.ops.aten.mm.default(div_15, permute_10);  div_15 = None
        convert_element_type_80: "f32[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_5: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.scatter.value(full_default_23, 1, where_28, -1.0);  where_28 = None
        where_29: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_65, div_10, full_default_3);  ne_65 = None
        mul_271: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(scatter_5, where_29);  scatter_5 = where_29 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_3: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_3, 30.0);  mm_3 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_3: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_3);  div_3 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_5: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_3, torch.float32);  tanh_3 = None
        mul_tensor_10: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1)
        sub_tensor_5: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_5, 30.0);  sub_tensor_5 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_75: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_11, log_3);  mul_tensor_11 = log_3 = None
        exp_14: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
        sum_33: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [1], True)
        mul_272: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(exp_14, sum_33);  exp_14 = sum_33 = None
        sub_143: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_81: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(sub_143, torch.bfloat16);  sub_143 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_273: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 30.0);  convert_element_type_81 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_82: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_273, torch.float32);  mul_273 = None
        mul_274: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, convert_element_type_default_5);  convert_element_type_default_5 = None
        sub_144: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(1, mul_274);  mul_274 = None
        mul_275: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_82, sub_144);  convert_element_type_82 = sub_144 = None
        convert_element_type_84: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_275, torch.bfloat16);  mul_275 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_16: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_84, 30.0);  convert_element_type_84 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_14: "bf16[((s3 + 8)//9), 2048]" = torch.ops.aten.mm.default(div_16, permute_10);  div_16 = None
        convert_element_type_87: "f32[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_6: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.scatter.value(full_default_23, 1, where_30, -1.0);  where_30 = None
        where_31: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_67, div_10, full_default_3);  ne_67 = None
        mul_276: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(scatter_6, where_31);  scatter_6 = where_31 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_2: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_2, 30.0);  mm_2 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_2: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_2);  div_2 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_6: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_2, torch.float32);  tanh_2 = None
        mul_tensor_12: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1)
        sub_tensor_6: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = amax_default_6 = None
        mul_tensor_13: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_6, 30.0);  sub_tensor_6 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_64: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_13, log_2);  mul_tensor_13 = log_2 = None
        exp_15: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_34: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [1], True)
        mul_277: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(exp_15, sum_34);  exp_15 = sum_34 = None
        sub_145: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_88: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(sub_145, torch.bfloat16);  sub_145 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_278: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 30.0);  convert_element_type_88 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_89: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_278, torch.float32);  mul_278 = None
        mul_279: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, convert_element_type_default_6);  convert_element_type_default_6 = None
        sub_146: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(1, mul_279);  mul_279 = None
        mul_280: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_89, sub_146);  convert_element_type_89 = sub_146 = None
        convert_element_type_91: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_17: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_91, 30.0);  convert_element_type_91 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_15: "bf16[((s3 + 8)//9), 2048]" = torch.ops.aten.mm.default(div_17, permute_10);  div_17 = None
        convert_element_type_94: "f32[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_7: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.scatter.value(full_default_23, 1, where_32, -1.0);  where_32 = None
        where_33: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_69, div_10, full_default_3);  ne_69 = None
        mul_281: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(scatter_7, where_33);  scatter_7 = where_33 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_1: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm_1, 30.0);  mm_1 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh_1: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div_1);  div_1 = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_7: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh_1, torch.float32);  tanh_1 = None
        mul_tensor_14: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1)
        sub_tensor_7: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = amax_default_7 = None
        mul_tensor_15: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_7, 30.0);  sub_tensor_7 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_53: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_15, log_1);  mul_tensor_15 = log_1 = None
        exp_16: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        sum_35: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [1], True)
        mul_282: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(exp_16, sum_35);  exp_16 = sum_35 = None
        sub_147: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_95: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(sub_147, torch.bfloat16);  sub_147 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_283: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 30.0);  convert_element_type_95 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_96: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_283, torch.float32);  mul_283 = None
        mul_284: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, convert_element_type_default_7);  convert_element_type_default_7 = None
        sub_148: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(1, mul_284);  mul_284 = None
        mul_285: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_96, sub_148);  convert_element_type_96 = sub_148 = None
        convert_element_type_98: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_285, torch.bfloat16);  mul_285 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_18: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_98, 30.0);  convert_element_type_98 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_16: "bf16[((s3 + 8)//9), 2048]" = torch.ops.aten.mm.default(div_18, permute_10);  div_18 = None
        convert_element_type_101: "f32[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        scatter_8: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.scatter.value(full_default_23, 1, where_34, -1.0);  full_default_23 = where_34 = None
        where_35: "f32[((s0 + 8)//9), 1]" = torch.ops.aten.where.self(ne_71, div_10, full_default_3);  ne_71 = div_10 = full_default_3 = None
        mul_286: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(scatter_8, where_35);  scatter_8 = where_35 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(mm, 30.0);  mm = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        tanh: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.tanh.default(div);  div = None
        
        # No stacktrace found for following nodes
        convert_element_type_default_8: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_tensor_16: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1)
        sub_tensor_8: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_16, amax_default_8);  mul_tensor_16 = amax_default_8 = None
        mul_tensor_17: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(sub_tensor_8, 30.0);  sub_tensor_8 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/patch_torch_functions.py:164 in cross_entropy, code: return torch._C._nn.cross_entropy_loss(
        sub_42: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_tensor_17, log);  mul_tensor_17 = log = None
        exp_17: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_36: "f32[((s3 + 8)//9), 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [1], True)
        mul_287: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(exp_17, sum_36);  exp_17 = sum_36 = None
        sub_149: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:554 in _unsloth_compiled_fused_ce_loss_function, code: input  = _shift_logits.float().contiguous(),
        convert_element_type_102: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(sub_149, torch.bfloat16);  sub_149 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:551 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits * logit_softcapping
        mul_288: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 30.0);  convert_element_type_102 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:550 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.tanh(_shift_logits)
        convert_element_type_103: "f32[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_288, torch.float32);  mul_288 = None
        mul_289: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, convert_element_type_default_8);  convert_element_type_default_8 = None
        sub_150: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.sub.Tensor(1, mul_289);  mul_289 = None
        mul_290: "f32[((s3 + 8)//9), 262400]" = torch.ops.aten.mul.Tensor(convert_element_type_103, sub_150);  convert_element_type_103 = sub_150 = None
        convert_element_type_105: "bf16[((s3 + 8)//9), 262400]" = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:549 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = _shift_logits / logit_softcapping
        div_19: "bf16[((s3 + 8)//9), 262400]" = torch.ops.aten.div.Tensor(convert_element_type_105, 30.0);  convert_element_type_105 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:537 in _unsloth_compiled_fused_ce_loss_function, code: _shift_logits = torch.nn.functional.linear(
        mm_17: "bf16[((s3 + 8)//9), 2048]" = torch.ops.aten.mm.default(div_19, permute_10);  div_19 = permute_10 = None
        convert_element_type_108: "f32[((s3 + 8)//9), 2048]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:530 in _unsloth_compiled_fused_ce_loss_function, code: __shift_states = torch.chunk(hidden_states, n_chunks, dim = 0)
        cat: "f32[s3, 2048]" = torch.ops.aten.cat.default([convert_element_type_108, convert_element_type_101, convert_element_type_94, convert_element_type_87, convert_element_type_80, convert_element_type_73, convert_element_type_66, convert_element_type_59, convert_element_type_52]);  convert_element_type_108 = convert_element_type_101 = convert_element_type_94 = convert_element_type_87 = convert_element_type_80 = convert_element_type_73 = convert_element_type_66 = convert_element_type_59 = convert_element_type_52 = None
        
         # File: /usr/local/lib/python3.10/dist-packages/unsloth_zoo/loss_utils.py:528 in _unsloth_compiled_fused_ce_loss_function, code: hidden_states = hidden_states.view(-1, hd)
        view_11: "f32[1, s3, 2048]" = torch.ops.aten.view.default(cat, [1, primals_6, 2048]);  cat = primals_6 = None
        return (None, None, None, None, None, None, view_11, None)
        