
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.verbose = False
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._dynamo.config.traceable_tensor_subclasses = set()
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.allowed_functions_module_string_ignorelist = {'torch._prims', 'torch._refs', 'torch.distributions', 'torch.testing', 'torch._decomp'}
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.optimize_ddp = True
torch._dynamo.config.do_not_emit_runtime_asserts = True
torch._dynamo.config._ddp_optimization_mode = ['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
torch._dynamo.config.numpy_default_float = 'float32'
torch._dynamo.config.inline_inbuilt_nn_modules = True
torch._dynamo.config._save_config_ignore = {'skipfiles_inline_module_allowlist', 'constant_functions', 'repro_level', 'repro_after'}
torch._dynamo.config.reorderable_logging_functions = set()
torch._dynamo.config.ignore_logger_methods = set()
torch._dynamo.config._autograd_backward_strict_mode_banned_ops = ['stride', 'requires_grad', 'storage_offset', 'layout', 'data', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference', 'is_ipu', 'is_leaf', 'is_maia', 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared', 'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu']
torch._dynamo.config.compiled_autograd = False
torch._dynamo.config.compiled_autograd_kwargs_override = {}
torch._inductor.config.debug = False
torch._inductor.config.disable_progress = True
torch._inductor.config.verbose_progress = False
torch._inductor.config.dce = True
torch._inductor.config.memory_planning = True
torch._inductor.config.memory_pool = 'none'
torch._inductor.config.epilogue_fusion = True
torch._inductor.config.efficient_conv_bn_eval_fx_passes = True
torch._inductor.config.group_fusion = False
torch._inductor.config.pre_grad_fusion_options = {}
torch._inductor.config.post_grad_fusion_options = {}
torch._inductor.config.dynamic_scale_rblock = True
torch._inductor.config.fx_passes_numeric_check = {'pre_grad': False, 'precision': 0.0001, 'num_iterations': 1, 'requires_optimizer': True}
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ['reorder_compute_for_overlap', 'sink_waits', 'raise_comms']
torch._inductor.config.max_autotune = False
torch._inductor.config.max_autotune_pointwise = False
torch._inductor.config.max_autotune_gemm = False
torch._inductor.config.max_autotune_gemm_backends = 'ATEN,TRITON,CPP'
torch._inductor.config.autotune_fallback_to_aten = True
torch._inductor.config.autotune_multi_device = True
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.aggressive_fusion = False
torch._inductor.config.combo_kernels = False
torch._inductor.config.benchmark_combo_kernel = False
torch._inductor.config.combo_kernel_foreach_dynamic_shapes = False
torch._inductor.config.emulate_precision_casts = False
torch._inductor.config._fuse_ddp_communication_passes = ['fuse_ddp_with_concat_op', 'schedule_comm_wait']
torch._inductor.config.shape_padding = True
torch._inductor.config.freezing = False
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cooperative_reductions = False
torch._inductor.config.triton.multi_kernel = 0
torch._inductor.config.triton.use_block_ptr = False
torch._inductor.config.triton.enable_persistent_tma_matmul = False
torch._inductor.config.aot_inductor.metadata = {}
torch._inductor.config.aot_inductor.presets = {}
torch._inductor.config.cuda.compile_opt_level = '-O1'
torch._inductor.config.cuda.enable_cuda_lto = True
torch._inductor.config.cuda.use_fast_math = True
torch._inductor.config.rocm.arch = []
torch._inductor.config.rocm.ck_supported_arch = ['gfx90a', 'gfx940', 'gfx941', 'gfx942']
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.graph_diagram = False
torch._inductor.config._save_config_ignore = ['trace.upload_tar', 'joint_custom_pre_pass', 'joint_custom_post_pass', 'pre_grad_custom_pass']
torch._inductor.config._cache_config_ignore_prefix = ['trace', 'cuda.cutlass_dir', 'worker_start_method', 'compile_threads', 'post_grad_custom_post_pass', 'post_grad_custom_pre_pass', 'always_complex_memory_overlap_TESTING_ONLY']
torch._inductor.config.external_matmul = []
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.6.0+cu124
# torch cuda version: 12.4
# torch git version: 2236df1770800ffea5697b11b0bb0d910b2e59e1


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA RTX A4000 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8):
        empty = torch.ops.aten.empty.memory_format([1, primals_2], dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute = torch.ops.aten.permute.default(empty, [0, 1]);  empty = None
        slice_1 = torch.ops.aten.slice.Tensor(primals_3, 1, 1, 9223372036854775807);  primals_3 = None
        slice_scatter = torch.ops.aten.slice_scatter.default(permute, slice_1, 1, 0, -1);  permute = slice_1 = None
        slice_5 = torch.ops.aten.slice.Tensor(primals_5, 1, 1, 9223372036854775807);  primals_5 = None
        eq_14 = torch.ops.aten.eq.Scalar(slice_5, 0);  slice_5 = None
        full_default = torch.ops.aten.full.default([], -100, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        slice_6 = torch.ops.aten.slice.Tensor(slice_scatter, 1, 0, -1)
        index_put = torch.ops.aten.index_put.default(slice_6, [eq_14], full_default);  slice_6 = eq_14 = None
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(slice_scatter, index_put, 1, 0, -1);  slice_scatter = index_put = None
        select_1 = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        copy_1 = torch.ops.aten.copy.default(select_1, full_default);  select_1 = full_default = None
        select_scatter = torch.ops.aten.select_scatter.default(slice_scatter_1, copy_1, 1, -1);  slice_scatter_1 = copy_1 = None
        view_1 = torch.ops.aten.view.default(primals_7, [-1, 2048]);  primals_7 = None
        add_31 = primals_6 + 9
        sub_14 = add_31 - 1;  add_31 = None
        floordiv = sub_14 // 9;  sub_14 = None
        split = torch.ops.aten.split.Tensor(view_1, floordiv);  view_1 = floordiv = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6]
        getitem_7 = split[7]
        getitem_8 = split[8];  split = None
        sym_size_int_8 = torch.ops.aten.sym_size.int(getitem_8, 0)
        add_59 = primals_2 + 9
        sub_24 = add_59 - 1;  add_59 = None
        floordiv_1 = sub_24 // 9;  sub_24 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(getitem, torch.bfloat16);  getitem = None
        permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        mm = torch.ops.aten.mm.default(convert_element_type, permute_1);  convert_element_type = None
        div = torch.ops.aten.div.Tensor(mm, 30.0)
        tanh = torch.ops.aten.tanh.default(div);  div = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1);  convert_element_type_default_8 = None
        amax_default_8 = torch.ops.aten.amax.default(mul_tensor_16, [1], True)
        sub_tensor_8 = torch.ops.aten.sub.Tensor(mul_tensor_16, amax_default_8);  mul_tensor_16 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(sub_tensor_8, 30.0);  sub_tensor_8 = None
        exp = torch.ops.aten.exp.default(mul_tensor_17)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_tensor_17, log);  mul_tensor_17 = None
        view_2 = torch.ops.aten.view.default(select_scatter, [-1]);  select_scatter = None
        split_2 = torch.ops.aten.split.Tensor(view_2, floordiv_1);  view_2 = floordiv_1 = None
        getitem_18 = split_2[0]
        ne_4 = torch.ops.aten.ne.Scalar(getitem_18, -100)
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne_4, getitem_18, full_default_2)
        unsqueeze = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(sub_42, 1, unsqueeze);  sub_42 = unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne_4, neg, full_default_3);  ne_4 = neg = None
        sum_3 = torch.ops.aten.sum.default(where_1);  where_1 = None
        add_105 = torch.ops.aten.add.Tensor(sum_3, 0.0);  sum_3 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(getitem_1, torch.bfloat16);  getitem_1 = None
        mm_1 = torch.ops.aten.mm.default(convert_element_type_5, permute_1);  convert_element_type_5 = None
        div_1 = torch.ops.aten.div.Tensor(mm_1, 30.0)
        tanh_1 = torch.ops.aten.tanh.default(div_1);  div_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(tanh_1, torch.float32);  tanh_1 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1);  convert_element_type_default_7 = None
        amax_default_7 = torch.ops.aten.amax.default(mul_tensor_14, [1], True)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_7, 30.0);  sub_tensor_7 = None
        exp_1 = torch.ops.aten.exp.default(mul_tensor_15)
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log_1 = torch.ops.aten.log.default(sum_4);  sum_4 = None
        sub_53 = torch.ops.aten.sub.Tensor(mul_tensor_15, log_1);  mul_tensor_15 = None
        getitem_28 = split_2[1]
        ne_10 = torch.ops.aten.ne.Scalar(getitem_28, -100)
        where_2 = torch.ops.aten.where.self(ne_10, getitem_28, full_default_2)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
        gather_1 = torch.ops.aten.gather.default(sub_53, 1, unsqueeze_1);  sub_53 = unsqueeze_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        where_3 = torch.ops.aten.where.self(ne_10, neg_1, full_default_3);  ne_10 = neg_1 = None
        sum_6 = torch.ops.aten.sum.default(where_3);  where_3 = None
        add_133 = torch.ops.aten.add.Tensor(add_105, sum_6);  add_105 = sum_6 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(getitem_2, torch.bfloat16);  getitem_2 = None
        mm_2 = torch.ops.aten.mm.default(convert_element_type_10, permute_1);  convert_element_type_10 = None
        div_2 = torch.ops.aten.div.Tensor(mm_2, 30.0)
        tanh_2 = torch.ops.aten.tanh.default(div_2);  div_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(tanh_2, torch.float32);  tanh_2 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1);  convert_element_type_default_6 = None
        amax_default_6 = torch.ops.aten.amax.default(mul_tensor_12, [1], True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_6, 30.0);  sub_tensor_6 = None
        exp_2 = torch.ops.aten.exp.default(mul_tensor_13)
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_2, [1], True);  exp_2 = None
        log_2 = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_64 = torch.ops.aten.sub.Tensor(mul_tensor_13, log_2);  mul_tensor_13 = None
        getitem_38 = split_2[2]
        ne_16 = torch.ops.aten.ne.Scalar(getitem_38, -100)
        where_4 = torch.ops.aten.where.self(ne_16, getitem_38, full_default_2)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(where_4, 1);  where_4 = None
        gather_2 = torch.ops.aten.gather.default(sub_64, 1, unsqueeze_2);  sub_64 = unsqueeze_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(gather_2, 1);  gather_2 = None
        neg_2 = torch.ops.aten.neg.default(squeeze_2);  squeeze_2 = None
        where_5 = torch.ops.aten.where.self(ne_16, neg_2, full_default_3);  ne_16 = neg_2 = None
        sum_9 = torch.ops.aten.sum.default(where_5);  where_5 = None
        add_161 = torch.ops.aten.add.Tensor(add_133, sum_9);  add_133 = sum_9 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(getitem_3, torch.bfloat16);  getitem_3 = None
        mm_3 = torch.ops.aten.mm.default(convert_element_type_15, permute_1);  convert_element_type_15 = None
        div_3 = torch.ops.aten.div.Tensor(mm_3, 30.0)
        tanh_3 = torch.ops.aten.tanh.default(div_3);  div_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(tanh_3, torch.float32);  tanh_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1);  convert_element_type_default_5 = None
        amax_default_5 = torch.ops.aten.amax.default(mul_tensor_10, [1], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, 30.0);  sub_tensor_5 = None
        exp_3 = torch.ops.aten.exp.default(mul_tensor_11)
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_3, [1], True);  exp_3 = None
        log_3 = torch.ops.aten.log.default(sum_10);  sum_10 = None
        sub_75 = torch.ops.aten.sub.Tensor(mul_tensor_11, log_3);  mul_tensor_11 = None
        getitem_48 = split_2[3]
        ne_22 = torch.ops.aten.ne.Scalar(getitem_48, -100)
        where_6 = torch.ops.aten.where.self(ne_22, getitem_48, full_default_2)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather_3 = torch.ops.aten.gather.default(sub_75, 1, unsqueeze_3);  sub_75 = unsqueeze_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(gather_3, 1);  gather_3 = None
        neg_3 = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        where_7 = torch.ops.aten.where.self(ne_22, neg_3, full_default_3);  ne_22 = neg_3 = None
        sum_12 = torch.ops.aten.sum.default(where_7);  where_7 = None
        add_189 = torch.ops.aten.add.Tensor(add_161, sum_12);  add_161 = sum_12 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(getitem_4, torch.bfloat16);  getitem_4 = None
        mm_4 = torch.ops.aten.mm.default(convert_element_type_20, permute_1);  convert_element_type_20 = None
        div_4 = torch.ops.aten.div.Tensor(mm_4, 30.0)
        tanh_4 = torch.ops.aten.tanh.default(div_4);  div_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(tanh_4, torch.float32);  tanh_4 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1);  convert_element_type_default_4 = None
        amax_default_4 = torch.ops.aten.amax.default(mul_tensor_8, [1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_4, 30.0);  sub_tensor_4 = None
        exp_4 = torch.ops.aten.exp.default(mul_tensor_9)
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_4, [1], True);  exp_4 = None
        log_4 = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_86 = torch.ops.aten.sub.Tensor(mul_tensor_9, log_4);  mul_tensor_9 = None
        getitem_58 = split_2[4]
        ne_28 = torch.ops.aten.ne.Scalar(getitem_58, -100)
        where_8 = torch.ops.aten.where.self(ne_28, getitem_58, full_default_2)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(where_8, 1);  where_8 = None
        gather_4 = torch.ops.aten.gather.default(sub_86, 1, unsqueeze_4);  sub_86 = unsqueeze_4 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(gather_4, 1);  gather_4 = None
        neg_4 = torch.ops.aten.neg.default(squeeze_4);  squeeze_4 = None
        where_9 = torch.ops.aten.where.self(ne_28, neg_4, full_default_3);  ne_28 = neg_4 = None
        sum_15 = torch.ops.aten.sum.default(where_9);  where_9 = None
        add_217 = torch.ops.aten.add.Tensor(add_189, sum_15);  add_189 = sum_15 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(getitem_5, torch.bfloat16);  getitem_5 = None
        mm_5 = torch.ops.aten.mm.default(convert_element_type_25, permute_1);  convert_element_type_25 = None
        div_5 = torch.ops.aten.div.Tensor(mm_5, 30.0)
        tanh_5 = torch.ops.aten.tanh.default(div_5);  div_5 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(tanh_5, torch.float32);  tanh_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        amax_default_3 = torch.ops.aten.amax.default(mul_tensor_6, [1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, 30.0);  sub_tensor_3 = None
        exp_5 = torch.ops.aten.exp.default(mul_tensor_7)
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_5, [1], True);  exp_5 = None
        log_5 = torch.ops.aten.log.default(sum_16);  sum_16 = None
        sub_97 = torch.ops.aten.sub.Tensor(mul_tensor_7, log_5);  mul_tensor_7 = None
        getitem_68 = split_2[5]
        ne_34 = torch.ops.aten.ne.Scalar(getitem_68, -100)
        where_10 = torch.ops.aten.where.self(ne_34, getitem_68, full_default_2)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(where_10, 1);  where_10 = None
        gather_5 = torch.ops.aten.gather.default(sub_97, 1, unsqueeze_5);  sub_97 = unsqueeze_5 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(gather_5, 1);  gather_5 = None
        neg_5 = torch.ops.aten.neg.default(squeeze_5);  squeeze_5 = None
        where_11 = torch.ops.aten.where.self(ne_34, neg_5, full_default_3);  ne_34 = neg_5 = None
        sum_18 = torch.ops.aten.sum.default(where_11);  where_11 = None
        add_245 = torch.ops.aten.add.Tensor(add_217, sum_18);  add_217 = sum_18 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(getitem_6, torch.bfloat16);  getitem_6 = None
        mm_6 = torch.ops.aten.mm.default(convert_element_type_30, permute_1);  convert_element_type_30 = None
        div_6 = torch.ops.aten.div.Tensor(mm_6, 30.0)
        tanh_6 = torch.ops.aten.tanh.default(div_6);  div_6 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(tanh_6, torch.float32);  tanh_6 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        amax_default_2 = torch.ops.aten.amax.default(mul_tensor_4, [1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, 30.0);  sub_tensor_2 = None
        exp_6 = torch.ops.aten.exp.default(mul_tensor_5)
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log_6 = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_108 = torch.ops.aten.sub.Tensor(mul_tensor_5, log_6);  mul_tensor_5 = None
        getitem_78 = split_2[6]
        ne_40 = torch.ops.aten.ne.Scalar(getitem_78, -100)
        where_12 = torch.ops.aten.where.self(ne_40, getitem_78, full_default_2)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_6 = torch.ops.aten.gather.default(sub_108, 1, unsqueeze_6);  sub_108 = unsqueeze_6 = None
        squeeze_6 = torch.ops.aten.squeeze.dim(gather_6, 1);  gather_6 = None
        neg_6 = torch.ops.aten.neg.default(squeeze_6);  squeeze_6 = None
        where_13 = torch.ops.aten.where.self(ne_40, neg_6, full_default_3);  ne_40 = neg_6 = None
        sum_21 = torch.ops.aten.sum.default(where_13);  where_13 = None
        add_273 = torch.ops.aten.add.Tensor(add_245, sum_21);  add_245 = sum_21 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(getitem_7, torch.bfloat16);  getitem_7 = None
        mm_7 = torch.ops.aten.mm.default(convert_element_type_35, permute_1);  convert_element_type_35 = None
        div_7 = torch.ops.aten.div.Tensor(mm_7, 30.0)
        tanh_7 = torch.ops.aten.tanh.default(div_7);  div_7 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(tanh_7, torch.float32);  tanh_7 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default_1 = torch.ops.aten.amax.default(mul_tensor_2, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, 30.0);  sub_tensor_1 = None
        exp_7 = torch.ops.aten.exp.default(mul_tensor_3)
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_7, [1], True);  exp_7 = None
        log_7 = torch.ops.aten.log.default(sum_22);  sum_22 = None
        sub_119 = torch.ops.aten.sub.Tensor(mul_tensor_3, log_7);  mul_tensor_3 = None
        getitem_88 = split_2[7]
        ne_46 = torch.ops.aten.ne.Scalar(getitem_88, -100)
        where_14 = torch.ops.aten.where.self(ne_46, getitem_88, full_default_2)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(where_14, 1);  where_14 = None
        gather_7 = torch.ops.aten.gather.default(sub_119, 1, unsqueeze_7);  sub_119 = unsqueeze_7 = None
        squeeze_7 = torch.ops.aten.squeeze.dim(gather_7, 1);  gather_7 = None
        neg_7 = torch.ops.aten.neg.default(squeeze_7);  squeeze_7 = None
        where_15 = torch.ops.aten.where.self(ne_46, neg_7, full_default_3);  ne_46 = neg_7 = None
        sum_24 = torch.ops.aten.sum.default(where_15);  where_15 = None
        add_301 = torch.ops.aten.add.Tensor(add_273, sum_24);  add_273 = sum_24 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(getitem_8, torch.bfloat16);  getitem_8 = None
        mm_8 = torch.ops.aten.mm.default(convert_element_type_40, permute_1);  convert_element_type_40 = None
        div_8 = torch.ops.aten.div.Tensor(mm_8, 30.0)
        tanh_8 = torch.ops.aten.tanh.default(div_8);  div_8 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(tanh_8, torch.float32);  tanh_8 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 1);  convert_element_type_default = None
        amax_default = torch.ops.aten.amax.default(mul_tensor, [1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, 30.0);  sub_tensor = None
        exp_8 = torch.ops.aten.exp.default(mul_tensor_1)
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_8, [1], True);  exp_8 = None
        log_8 = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_130 = torch.ops.aten.sub.Tensor(mul_tensor_1, log_8);  mul_tensor_1 = None
        getitem_98 = split_2[8];  split_2 = None
        ne_52 = torch.ops.aten.ne.Scalar(getitem_98, -100)
        where_16 = torch.ops.aten.where.self(ne_52, getitem_98, full_default_2)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(where_16, 1);  where_16 = None
        gather_8 = torch.ops.aten.gather.default(sub_130, 1, unsqueeze_8);  sub_130 = unsqueeze_8 = None
        squeeze_8 = torch.ops.aten.squeeze.dim(gather_8, 1);  gather_8 = None
        neg_8 = torch.ops.aten.neg.default(squeeze_8);  squeeze_8 = None
        where_17 = torch.ops.aten.where.self(ne_52, neg_8, full_default_3);  ne_52 = neg_8 = full_default_3 = None
        sum_27 = torch.ops.aten.sum.default(where_17);  where_17 = None
        add_329 = torch.ops.aten.add.Tensor(add_301, sum_27);  add_301 = sum_27 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(primals_8, torch.float32);  primals_8 = None
        div_9 = torch.ops.aten.div.Tensor(add_329, convert_element_type_45);  add_329 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(getitem_98, 1);  getitem_98 = None
        ne_55 = torch.ops.aten.ne.Scalar(unsqueeze_9, -100)
        where_18 = torch.ops.aten.where.self(ne_55, unsqueeze_9, full_default_2);  unsqueeze_9 = None
        permute_10 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(getitem_88, 1);  getitem_88 = None
        ne_57 = torch.ops.aten.ne.Scalar(unsqueeze_10, -100)
        where_20 = torch.ops.aten.where.self(ne_57, unsqueeze_10, full_default_2);  unsqueeze_10 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(getitem_78, 1);  getitem_78 = None
        ne_59 = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        where_22 = torch.ops.aten.where.self(ne_59, unsqueeze_11, full_default_2);  unsqueeze_11 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(getitem_68, 1);  getitem_68 = None
        ne_61 = torch.ops.aten.ne.Scalar(unsqueeze_12, -100)
        where_24 = torch.ops.aten.where.self(ne_61, unsqueeze_12, full_default_2);  unsqueeze_12 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(getitem_58, 1);  getitem_58 = None
        ne_63 = torch.ops.aten.ne.Scalar(unsqueeze_13, -100)
        where_26 = torch.ops.aten.where.self(ne_63, unsqueeze_13, full_default_2);  unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(getitem_48, 1);  getitem_48 = None
        ne_65 = torch.ops.aten.ne.Scalar(unsqueeze_14, -100)
        where_28 = torch.ops.aten.where.self(ne_65, unsqueeze_14, full_default_2);  unsqueeze_14 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(getitem_38, 1);  getitem_38 = None
        ne_67 = torch.ops.aten.ne.Scalar(unsqueeze_15, -100)
        where_30 = torch.ops.aten.where.self(ne_67, unsqueeze_15, full_default_2);  unsqueeze_15 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(getitem_28, 1);  getitem_28 = None
        ne_69 = torch.ops.aten.ne.Scalar(unsqueeze_16, -100)
        where_32 = torch.ops.aten.where.self(ne_69, unsqueeze_16, full_default_2);  unsqueeze_16 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(getitem_18, 1);  getitem_18 = None
        ne_71 = torch.ops.aten.ne.Scalar(unsqueeze_17, -100)
        where_34 = torch.ops.aten.where.self(ne_71, unsqueeze_17, full_default_2);  unsqueeze_17 = full_default_2 = None
        return (div_9, mm, amax_default_8, log, mm_1, amax_default_7, log_1, mm_2, amax_default_6, log_2, mm_3, amax_default_5, log_3, mm_4, amax_default_4, log_4, mm_5, amax_default_3, log_5, mm_6, amax_default_2, log_6, mm_7, amax_default_1, log_7, mm_8, amax_default, log_8, convert_element_type_45, ne_55, where_18, permute_10, ne_57, where_20, ne_59, where_22, ne_61, where_24, ne_63, where_26, ne_65, where_28, ne_67, where_30, ne_69, where_32, ne_71, where_34, primals_6, primals_2, sym_size_int_8)
        
def load_args(reader):
    buf0 = reader.storage(None, 1074790400, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (262400, 2048), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    reader.symint(533)  # primals_2
    buf1 = reader.storage(None, 8*s0, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, s0), dtype=torch.int64, is_leaf=True)  # primals_3
    reader.symint(533)  # primals_4
    buf2 = reader.storage(None, 8*s1, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1, s1), dtype=torch.int64, is_leaf=True)  # primals_5
    reader.symint(533)  # primals_6
    buf3 = reader.storage(None, 8192*s2, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1, s3, 2048), is_leaf=True)  # primals_7
    buf4 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf4, (), dtype=torch.int64, is_leaf=True)  # primals_8
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)