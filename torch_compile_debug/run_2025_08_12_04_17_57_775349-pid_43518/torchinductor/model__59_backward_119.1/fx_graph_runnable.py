
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

    
    
    def forward(self, primals_6, primals_2, sym_size_int_8, mm, amax_default_8, log, mm_1, amax_default_7, log_1, mm_2, amax_default_6, log_2, mm_3, amax_default_5, log_3, mm_4, amax_default_4, log_4, mm_5, amax_default_3, log_5, mm_6, amax_default_2, log_6, mm_7, amax_default_1, log_7, mm_8, amax_default, log_8, convert_element_type_45, ne_55, where_18, permute_10, ne_57, where_20, ne_59, where_22, ne_61, where_24, ne_63, where_26, ne_65, where_28, ne_67, where_30, ne_69, where_32, ne_71, where_34, tangents_1):
        div_10 = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_45);  tangents_1 = convert_element_type_45 = None
        full = torch.ops.aten.full.default([sym_size_int_8, 262400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  sym_size_int_8 = None
        scatter = torch.ops.aten.scatter.value(full, 1, where_18, -1.0);  full = where_18 = None
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19 = torch.ops.aten.where.self(ne_55, div_10, full_default_3);  ne_55 = None
        mul_246 = torch.ops.aten.mul.Tensor(scatter, where_19);  scatter = where_19 = None
        div_8 = torch.ops.aten.div.Tensor(mm_8, 30.0);  mm_8 = None
        tanh_8 = torch.ops.aten.tanh.default(div_8);  div_8 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(tanh_8, torch.float32);  tanh_8 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 1)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, 30.0);  sub_tensor = None
        sub_130 = torch.ops.aten.sub.Tensor(mul_tensor_1, log_8);  mul_tensor_1 = log_8 = None
        exp_9 = torch.ops.aten.exp.default(sub_130);  sub_130 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(mul_246, [1], True)
        mul_247 = torch.ops.aten.mul.Tensor(exp_9, sum_28);  exp_9 = sum_28 = None
        sub_133 = torch.ops.aten.sub.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(sub_133, torch.bfloat16);  sub_133 = None
        mul_248 = torch.ops.aten.mul.Tensor(convert_element_type_46, 30.0);  convert_element_type_46 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(mul_248, torch.float32);  mul_248 = None
        mul_249 = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default);  convert_element_type_default = None
        sub_134 = torch.ops.aten.sub.Tensor(1, mul_249);  mul_249 = None
        mul_250 = torch.ops.aten.mul.Tensor(convert_element_type_47, sub_134);  convert_element_type_47 = sub_134 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None
        div_11 = torch.ops.aten.div.Tensor(convert_element_type_49, 30.0);  convert_element_type_49 = None
        mm_9 = torch.ops.aten.mm.default(div_11, permute_10);  div_11 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        add_31 = primals_6 + 9
        sub_14 = add_31 - 1;  add_31 = None
        floordiv = sub_14 // 9;  sub_14 = None
        full_default_23 = torch.ops.aten.full.default([floordiv, 262400], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  floordiv = None
        scatter_1 = torch.ops.aten.scatter.value(full_default_23, 1, where_20, -1.0);  where_20 = None
        where_21 = torch.ops.aten.where.self(ne_57, div_10, full_default_3);  ne_57 = None
        mul_251 = torch.ops.aten.mul.Tensor(scatter_1, where_21);  scatter_1 = where_21 = None
        div_7 = torch.ops.aten.div.Tensor(mm_7, 30.0);  mm_7 = None
        tanh_7 = torch.ops.aten.tanh.default(div_7);  div_7 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(tanh_7, torch.float32);  tanh_7 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, 30.0);  sub_tensor_1 = None
        sub_119 = torch.ops.aten.sub.Tensor(mul_tensor_3, log_7);  mul_tensor_3 = log_7 = None
        exp_10 = torch.ops.aten.exp.default(sub_119);  sub_119 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(mul_251, [1], True)
        mul_252 = torch.ops.aten.mul.Tensor(exp_10, sum_29);  exp_10 = sum_29 = None
        sub_135 = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(sub_135, torch.bfloat16);  sub_135 = None
        mul_253 = torch.ops.aten.mul.Tensor(convert_element_type_53, 30.0);  convert_element_type_53 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(mul_253, torch.float32);  mul_253 = None
        mul_254 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default_1);  convert_element_type_default_1 = None
        sub_136 = torch.ops.aten.sub.Tensor(1, mul_254);  mul_254 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_54, sub_136);  convert_element_type_54 = sub_136 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(mul_255, torch.bfloat16);  mul_255 = None
        div_12 = torch.ops.aten.div.Tensor(convert_element_type_56, 30.0);  convert_element_type_56 = None
        mm_10 = torch.ops.aten.mm.default(div_12, permute_10);  div_12 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        scatter_2 = torch.ops.aten.scatter.value(full_default_23, 1, where_22, -1.0);  where_22 = None
        where_23 = torch.ops.aten.where.self(ne_59, div_10, full_default_3);  ne_59 = None
        mul_256 = torch.ops.aten.mul.Tensor(scatter_2, where_23);  scatter_2 = where_23 = None
        div_6 = torch.ops.aten.div.Tensor(mm_6, 30.0);  mm_6 = None
        tanh_6 = torch.ops.aten.tanh.default(div_6);  div_6 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(tanh_6, torch.float32);  tanh_6 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, 30.0);  sub_tensor_2 = None
        sub_108 = torch.ops.aten.sub.Tensor(mul_tensor_5, log_6);  mul_tensor_5 = log_6 = None
        exp_11 = torch.ops.aten.exp.default(sub_108);  sub_108 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(mul_256, [1], True)
        mul_257 = torch.ops.aten.mul.Tensor(exp_11, sum_30);  exp_11 = sum_30 = None
        sub_137 = torch.ops.aten.sub.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        convert_element_type_60 = torch.ops.prims.convert_element_type.default(sub_137, torch.bfloat16);  sub_137 = None
        mul_258 = torch.ops.aten.mul.Tensor(convert_element_type_60, 30.0);  convert_element_type_60 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(mul_258, torch.float32);  mul_258 = None
        mul_259 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default_2);  convert_element_type_default_2 = None
        sub_138 = torch.ops.aten.sub.Tensor(1, mul_259);  mul_259 = None
        mul_260 = torch.ops.aten.mul.Tensor(convert_element_type_61, sub_138);  convert_element_type_61 = sub_138 = None
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(mul_260, torch.bfloat16);  mul_260 = None
        div_13 = torch.ops.aten.div.Tensor(convert_element_type_63, 30.0);  convert_element_type_63 = None
        mm_11 = torch.ops.aten.mm.default(div_13, permute_10);  div_13 = None
        convert_element_type_66 = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        scatter_3 = torch.ops.aten.scatter.value(full_default_23, 1, where_24, -1.0);  where_24 = None
        where_25 = torch.ops.aten.where.self(ne_61, div_10, full_default_3);  ne_61 = None
        mul_261 = torch.ops.aten.mul.Tensor(scatter_3, where_25);  scatter_3 = where_25 = None
        div_5 = torch.ops.aten.div.Tensor(mm_5, 30.0);  mm_5 = None
        tanh_5 = torch.ops.aten.tanh.default(div_5);  div_5 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(tanh_5, torch.float32);  tanh_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, 30.0);  sub_tensor_3 = None
        sub_97 = torch.ops.aten.sub.Tensor(mul_tensor_7, log_5);  mul_tensor_7 = log_5 = None
        exp_12 = torch.ops.aten.exp.default(sub_97);  sub_97 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(mul_261, [1], True)
        mul_262 = torch.ops.aten.mul.Tensor(exp_12, sum_31);  exp_12 = sum_31 = None
        sub_139 = torch.ops.aten.sub.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(sub_139, torch.bfloat16);  sub_139 = None
        mul_263 = torch.ops.aten.mul.Tensor(convert_element_type_67, 30.0);  convert_element_type_67 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(mul_263, torch.float32);  mul_263 = None
        mul_264 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, convert_element_type_default_3);  convert_element_type_default_3 = None
        sub_140 = torch.ops.aten.sub.Tensor(1, mul_264);  mul_264 = None
        mul_265 = torch.ops.aten.mul.Tensor(convert_element_type_68, sub_140);  convert_element_type_68 = sub_140 = None
        convert_element_type_70 = torch.ops.prims.convert_element_type.default(mul_265, torch.bfloat16);  mul_265 = None
        div_14 = torch.ops.aten.div.Tensor(convert_element_type_70, 30.0);  convert_element_type_70 = None
        mm_12 = torch.ops.aten.mm.default(div_14, permute_10);  div_14 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        scatter_4 = torch.ops.aten.scatter.value(full_default_23, 1, where_26, -1.0);  where_26 = None
        where_27 = torch.ops.aten.where.self(ne_63, div_10, full_default_3);  ne_63 = None
        mul_266 = torch.ops.aten.mul.Tensor(scatter_4, where_27);  scatter_4 = where_27 = None
        div_4 = torch.ops.aten.div.Tensor(mm_4, 30.0);  mm_4 = None
        tanh_4 = torch.ops.aten.tanh.default(div_4);  div_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(tanh_4, torch.float32);  tanh_4 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_4, 30.0);  sub_tensor_4 = None
        sub_86 = torch.ops.aten.sub.Tensor(mul_tensor_9, log_4);  mul_tensor_9 = log_4 = None
        exp_13 = torch.ops.aten.exp.default(sub_86);  sub_86 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(mul_266, [1], True)
        mul_267 = torch.ops.aten.mul.Tensor(exp_13, sum_32);  exp_13 = sum_32 = None
        sub_141 = torch.ops.aten.sub.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
        convert_element_type_74 = torch.ops.prims.convert_element_type.default(sub_141, torch.bfloat16);  sub_141 = None
        mul_268 = torch.ops.aten.mul.Tensor(convert_element_type_74, 30.0);  convert_element_type_74 = None
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(mul_268, torch.float32);  mul_268 = None
        mul_269 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, convert_element_type_default_4);  convert_element_type_default_4 = None
        sub_142 = torch.ops.aten.sub.Tensor(1, mul_269);  mul_269 = None
        mul_270 = torch.ops.aten.mul.Tensor(convert_element_type_75, sub_142);  convert_element_type_75 = sub_142 = None
        convert_element_type_77 = torch.ops.prims.convert_element_type.default(mul_270, torch.bfloat16);  mul_270 = None
        div_15 = torch.ops.aten.div.Tensor(convert_element_type_77, 30.0);  convert_element_type_77 = None
        mm_13 = torch.ops.aten.mm.default(div_15, permute_10);  div_15 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        scatter_5 = torch.ops.aten.scatter.value(full_default_23, 1, where_28, -1.0);  where_28 = None
        where_29 = torch.ops.aten.where.self(ne_65, div_10, full_default_3);  ne_65 = None
        mul_271 = torch.ops.aten.mul.Tensor(scatter_5, where_29);  scatter_5 = where_29 = None
        div_3 = torch.ops.aten.div.Tensor(mm_3, 30.0);  mm_3 = None
        tanh_3 = torch.ops.aten.tanh.default(div_3);  div_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(tanh_3, torch.float32);  tanh_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, 30.0);  sub_tensor_5 = None
        sub_75 = torch.ops.aten.sub.Tensor(mul_tensor_11, log_3);  mul_tensor_11 = log_3 = None
        exp_14 = torch.ops.aten.exp.default(sub_75);  sub_75 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(mul_271, [1], True)
        mul_272 = torch.ops.aten.mul.Tensor(exp_14, sum_33);  exp_14 = sum_33 = None
        sub_143 = torch.ops.aten.sub.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
        convert_element_type_81 = torch.ops.prims.convert_element_type.default(sub_143, torch.bfloat16);  sub_143 = None
        mul_273 = torch.ops.aten.mul.Tensor(convert_element_type_81, 30.0);  convert_element_type_81 = None
        convert_element_type_82 = torch.ops.prims.convert_element_type.default(mul_273, torch.float32);  mul_273 = None
        mul_274 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, convert_element_type_default_5);  convert_element_type_default_5 = None
        sub_144 = torch.ops.aten.sub.Tensor(1, mul_274);  mul_274 = None
        mul_275 = torch.ops.aten.mul.Tensor(convert_element_type_82, sub_144);  convert_element_type_82 = sub_144 = None
        convert_element_type_84 = torch.ops.prims.convert_element_type.default(mul_275, torch.bfloat16);  mul_275 = None
        div_16 = torch.ops.aten.div.Tensor(convert_element_type_84, 30.0);  convert_element_type_84 = None
        mm_14 = torch.ops.aten.mm.default(div_16, permute_10);  div_16 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        scatter_6 = torch.ops.aten.scatter.value(full_default_23, 1, where_30, -1.0);  where_30 = None
        where_31 = torch.ops.aten.where.self(ne_67, div_10, full_default_3);  ne_67 = None
        mul_276 = torch.ops.aten.mul.Tensor(scatter_6, where_31);  scatter_6 = where_31 = None
        div_2 = torch.ops.aten.div.Tensor(mm_2, 30.0);  mm_2 = None
        tanh_2 = torch.ops.aten.tanh.default(div_2);  div_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(tanh_2, torch.float32);  tanh_2 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = amax_default_6 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_6, 30.0);  sub_tensor_6 = None
        sub_64 = torch.ops.aten.sub.Tensor(mul_tensor_13, log_2);  mul_tensor_13 = log_2 = None
        exp_15 = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(mul_276, [1], True)
        mul_277 = torch.ops.aten.mul.Tensor(exp_15, sum_34);  exp_15 = sum_34 = None
        sub_145 = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(sub_145, torch.bfloat16);  sub_145 = None
        mul_278 = torch.ops.aten.mul.Tensor(convert_element_type_88, 30.0);  convert_element_type_88 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(mul_278, torch.float32);  mul_278 = None
        mul_279 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, convert_element_type_default_6);  convert_element_type_default_6 = None
        sub_146 = torch.ops.aten.sub.Tensor(1, mul_279);  mul_279 = None
        mul_280 = torch.ops.aten.mul.Tensor(convert_element_type_89, sub_146);  convert_element_type_89 = sub_146 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None
        div_17 = torch.ops.aten.div.Tensor(convert_element_type_91, 30.0);  convert_element_type_91 = None
        mm_15 = torch.ops.aten.mm.default(div_17, permute_10);  div_17 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        scatter_7 = torch.ops.aten.scatter.value(full_default_23, 1, where_32, -1.0);  where_32 = None
        where_33 = torch.ops.aten.where.self(ne_69, div_10, full_default_3);  ne_69 = None
        mul_281 = torch.ops.aten.mul.Tensor(scatter_7, where_33);  scatter_7 = where_33 = None
        div_1 = torch.ops.aten.div.Tensor(mm_1, 30.0);  mm_1 = None
        tanh_1 = torch.ops.aten.tanh.default(div_1);  div_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(tanh_1, torch.float32);  tanh_1 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = amax_default_7 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_7, 30.0);  sub_tensor_7 = None
        sub_53 = torch.ops.aten.sub.Tensor(mul_tensor_15, log_1);  mul_tensor_15 = log_1 = None
        exp_16 = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(mul_281, [1], True)
        mul_282 = torch.ops.aten.mul.Tensor(exp_16, sum_35);  exp_16 = sum_35 = None
        sub_147 = torch.ops.aten.sub.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        convert_element_type_95 = torch.ops.prims.convert_element_type.default(sub_147, torch.bfloat16);  sub_147 = None
        mul_283 = torch.ops.aten.mul.Tensor(convert_element_type_95, 30.0);  convert_element_type_95 = None
        convert_element_type_96 = torch.ops.prims.convert_element_type.default(mul_283, torch.float32);  mul_283 = None
        mul_284 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, convert_element_type_default_7);  convert_element_type_default_7 = None
        sub_148 = torch.ops.aten.sub.Tensor(1, mul_284);  mul_284 = None
        mul_285 = torch.ops.aten.mul.Tensor(convert_element_type_96, sub_148);  convert_element_type_96 = sub_148 = None
        convert_element_type_98 = torch.ops.prims.convert_element_type.default(mul_285, torch.bfloat16);  mul_285 = None
        div_18 = torch.ops.aten.div.Tensor(convert_element_type_98, 30.0);  convert_element_type_98 = None
        mm_16 = torch.ops.aten.mm.default(div_18, permute_10);  div_18 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        scatter_8 = torch.ops.aten.scatter.value(full_default_23, 1, where_34, -1.0);  full_default_23 = where_34 = None
        where_35 = torch.ops.aten.where.self(ne_71, div_10, full_default_3);  ne_71 = div_10 = full_default_3 = None
        mul_286 = torch.ops.aten.mul.Tensor(scatter_8, where_35);  scatter_8 = where_35 = None
        div = torch.ops.aten.div.Tensor(mm, 30.0);  mm = None
        tanh = torch.ops.aten.tanh.default(div);  div = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1)
        sub_tensor_8 = torch.ops.aten.sub.Tensor(mul_tensor_16, amax_default_8);  mul_tensor_16 = amax_default_8 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(sub_tensor_8, 30.0);  sub_tensor_8 = None
        sub_42 = torch.ops.aten.sub.Tensor(mul_tensor_17, log);  mul_tensor_17 = log = None
        exp_17 = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(mul_286, [1], True)
        mul_287 = torch.ops.aten.mul.Tensor(exp_17, sum_36);  exp_17 = sum_36 = None
        sub_149 = torch.ops.aten.sub.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
        convert_element_type_102 = torch.ops.prims.convert_element_type.default(sub_149, torch.bfloat16);  sub_149 = None
        mul_288 = torch.ops.aten.mul.Tensor(convert_element_type_102, 30.0);  convert_element_type_102 = None
        convert_element_type_103 = torch.ops.prims.convert_element_type.default(mul_288, torch.float32);  mul_288 = None
        mul_289 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, convert_element_type_default_8);  convert_element_type_default_8 = None
        sub_150 = torch.ops.aten.sub.Tensor(1, mul_289);  mul_289 = None
        mul_290 = torch.ops.aten.mul.Tensor(convert_element_type_103, sub_150);  convert_element_type_103 = sub_150 = None
        convert_element_type_105 = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None
        div_19 = torch.ops.aten.div.Tensor(convert_element_type_105, 30.0);  convert_element_type_105 = None
        mm_17 = torch.ops.aten.mm.default(div_19, permute_10);  div_19 = permute_10 = None
        convert_element_type_108 = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        cat = torch.ops.aten.cat.default([convert_element_type_108, convert_element_type_101, convert_element_type_94, convert_element_type_87, convert_element_type_80, convert_element_type_73, convert_element_type_66, convert_element_type_59, convert_element_type_52]);  convert_element_type_108 = convert_element_type_101 = convert_element_type_94 = convert_element_type_87 = convert_element_type_80 = convert_element_type_73 = convert_element_type_66 = convert_element_type_59 = convert_element_type_52 = None
        view_11 = torch.ops.aten.view.default(cat, [1, primals_6, 2048]);  cat = primals_6 = None
        return (None, None, None, None, None, None, view_11, None)
        
def load_args(reader):
    reader.symint(533)  # primals_6
    reader.symint(533)  # primals_2
    reader.symint(53)  # sym_size_int_8
    buf0 = reader.storage(None, 524800*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (((s3 + 8)//9), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm
    buf1 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf1, (((s3 + 8)//9), 1), is_leaf=True)  # amax_default_8
    buf2 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf2, (((s3 + 8)//9), 1), is_leaf=True)  # log
    buf3 = reader.storage(None, 524800*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (((s3 + 8)//9), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm_1
    buf4 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf4, (((s3 + 8)//9), 1), is_leaf=True)  # amax_default_7
    buf5 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf5, (((s3 + 8)//9), 1), is_leaf=True)  # log_1
    buf6 = reader.storage(None, 524800*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (((s3 + 8)//9), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm_2
    buf7 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf7, (((s3 + 8)//9), 1), is_leaf=True)  # amax_default_6
    buf8 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf8, (((s3 + 8)//9), 1), is_leaf=True)  # log_2
    buf9 = reader.storage(None, 524800*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (((s3 + 8)//9), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm_3
    buf10 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf10, (((s3 + 8)//9), 1), is_leaf=True)  # amax_default_5
    buf11 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf11, (((s3 + 8)//9), 1), is_leaf=True)  # log_3
    buf12 = reader.storage(None, 524800*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (((s3 + 8)//9), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm_4
    buf13 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf13, (((s3 + 8)//9), 1), is_leaf=True)  # amax_default_4
    buf14 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf14, (((s3 + 8)//9), 1), is_leaf=True)  # log_4
    buf15 = reader.storage(None, 524800*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (((s3 + 8)//9), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm_5
    buf16 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf16, (((s3 + 8)//9), 1), is_leaf=True)  # amax_default_3
    buf17 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf17, (((s3 + 8)//9), 1), is_leaf=True)  # log_5
    buf18 = reader.storage(None, 524800*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf18, (((s3 + 8)//9), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm_6
    buf19 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf19, (((s3 + 8)//9), 1), is_leaf=True)  # amax_default_2
    buf20 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf20, (((s3 + 8)//9), 1), is_leaf=True)  # log_6
    buf21 = reader.storage(None, 524800*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf21, (((s3 + 8)//9), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm_7
    buf22 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf22, (((s3 + 8)//9), 1), is_leaf=True)  # amax_default_1
    buf23 = reader.storage(None, 4*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf23, (((s3 + 8)//9), 1), is_leaf=True)  # log_7
    buf24 = reader.storage(None, 524800*s3 - 4198400*(((s3 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf24, (s3 - 8*(((s3 + 8)//9)), 262400), dtype=torch.bfloat16, is_leaf=True)  # mm_8
    buf25 = reader.storage(None, 4*s3 - 32*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf25, (s3 - 8*(((s3 + 8)//9)), 1), is_leaf=True)  # amax_default
    buf26 = reader.storage(None, 4*s3 - 32*(((s3 + 8)//9)), device=device(type='cuda', index=0))
    reader.tensor(buf26, (s3 - 8*(((s3 + 8)//9)), 1), is_leaf=True)  # log_8
    buf27 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf27, (), is_leaf=True)  # convert_element_type_45
    buf28 = reader.storage(None, s0 - 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf28, (s0 - 8*(((s0 + 8)//9)), 1), dtype=torch.bool, is_leaf=True)  # ne_55
    buf29 = reader.storage(None, 8*s0 - 64*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf29, (s0 - 8*(((s0 + 8)//9)), 1), dtype=torch.int64, is_leaf=True)  # where_18
    buf30 = reader.storage(None, 1074790400, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf30, (262400, 2048), dtype=torch.bfloat16, is_leaf=True)  # permute_10
    buf31 = reader.storage(None, ((s0 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf31, (((s0 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_57
    buf32 = reader.storage(None, 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf32, (((s0 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_20
    buf33 = reader.storage(None, ((s0 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf33, (((s0 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_59
    buf34 = reader.storage(None, 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf34, (((s0 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_22
    buf35 = reader.storage(None, ((s0 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf35, (((s0 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_61
    buf36 = reader.storage(None, 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf36, (((s0 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_24
    buf37 = reader.storage(None, ((s0 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf37, (((s0 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_63
    buf38 = reader.storage(None, 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf38, (((s0 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_26
    buf39 = reader.storage(None, ((s0 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf39, (((s0 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_65
    buf40 = reader.storage(None, 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf40, (((s0 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_28
    buf41 = reader.storage(None, ((s0 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf41, (((s0 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_67
    buf42 = reader.storage(None, 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf42, (((s0 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_30
    buf43 = reader.storage(None, ((s0 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf43, (((s0 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_69
    buf44 = reader.storage(None, 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf44, (((s0 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_32
    buf45 = reader.storage(None, ((s0 + 8)//9), device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf45, (((s0 + 8)//9), 1), dtype=torch.bool, is_leaf=True)  # ne_71
    buf46 = reader.storage(None, 8*(((s0 + 8)//9)), device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf46, (((s0 + 8)//9), 1), dtype=torch.int64, is_leaf=True)  # where_34
    buf47 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf47, (), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)