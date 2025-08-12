# AOT ID: ['59_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/3y/c3ygfb2zluppsv34m7xs4ugrsf7vpapfam4zosql4jrl2rfwomq2.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_3
# Graph fragment:
#   %div_10 : [num_users=9] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_45), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [%sym_size_int_8, 262400], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_18, val: -1.0})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_19 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_55, %div_10, %full_default_3), kwargs = {})
#   %mul_246 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_19), kwargs = {})
#   %sum_28 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_246, [1], True), kwargs = {})
triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 524288},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 262400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp1 = r1
        tmp2 = tmp0 == tmp1
        tmp3 = -1.0
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp11 = tmp8 / tmp10
        tmp12 = tl.where(tmp6, tmp11, tmp4)
        tmp13 = tmp5 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/kg/ckgkatxdb4c54gi2zdhkfrybbmq3ur246ukq42nd3ngdsr77w5jd.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_33, _shift_logits_34, cross_entropy_loss_8], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
# Source node to ATen node mapping:
#   _shift_logits_33 => div_8
#   _shift_logits_34 => tanh_8
#   cross_entropy_loss => full_default_3
#   cross_entropy_loss_8 => sub_130
# Graph fragment:
#   %div_10 : [num_users=9] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_45), kwargs = {})
#   %scatter_upon_const_tensor : [num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [%sym_size_int_8, 262400], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_18, val: -1.0})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_19 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_55, %div_10, %full_default_3), kwargs = {})
#   %mul_246 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_19), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_8, 30.0), kwargs = {})
#   %tanh_8 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_8,), kwargs = {})
#   %convert_element_type_default : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tanh_8, torch.float32), kwargs = {})
#   %mul_tensor : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, 1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %mul_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor, 30.0), kwargs = {})
#   %sub_130 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %log_8), kwargs = {})
#   %exp_9 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_130,), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_9, %sum_28), kwargs = {})
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_246, %mul_247), kwargs = {})
#   %convert_element_type_46 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_133, torch.bfloat16), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_46, 30.0), kwargs = {})
#   %convert_element_type_47 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_248, torch.float32), kwargs = {})
#   %mul_249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default, %convert_element_type_default), kwargs = {})
#   %sub_134 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_249), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_47, %sub_134), kwargs = {})
#   %convert_element_type_49 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_250, torch.bfloat16), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_49, 30.0), kwargs = {})
triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_1 = async_compile.triton('triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 262400
    x0 = (xindex % 262400)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp14 = tl.load(in_ptr4 + (x2), xmask).to(tl.float32)
    tmp21 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp0 == tmp1
    tmp3 = -1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp11 = tmp8 / tmp10
    tmp12 = tl.where(tmp6, tmp11, tmp4)
    tmp13 = tmp5 * tmp12
    tmp15 = 0.03333333333333333
    tmp16 = tmp14 * tmp15
    tmp17 = libdevice.tanh(tmp16)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 - tmp21
    tmp23 = 30.0
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 - tmp25
    tmp27 = tl_math.exp(tmp26)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp13 - tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31 * tmp23
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp18 * tmp18
    tmp35 = tmp19 - tmp34
    tmp36 = tmp33 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 * tmp15
    tl.store(in_out_ptr0 + (x2), tmp38, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rw/crwvhygfepms5yn6lbi53ok35wjdfpxv2xeuswj7uxyrprk4jmuf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_23 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([%floordiv, 262400], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default_23, 1, %where_20, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_2 = async_compile.triton('triton_poi_fused_nll_loss_backward_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/dy/cdyh7fkmf2aua4fncfktucsvdzxgqpkwmxlcrauchnqo5udwe7ts.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_23 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([%floordiv, 262400], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default_23, 1, %where_20, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_3 = async_compile.triton('triton_poi_fused_nll_loss_backward_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_3', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 262400)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 262400")
    tmp2 = -1.0
    tl.store(out_ptr0 + (tmp0 + 262400*x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/br/cbremklc6ukxo6utzlu4iatgeelp3o5r4wbogrx5txpt22v2tnkp.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_3
# Graph fragment:
#   %div_10 : [num_users=9] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_45), kwargs = {})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_57, %div_10, %full_default_3), kwargs = {})
#   %mul_251 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_21), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_251, [1], True), kwargs = {})
triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4 = async_compile.triton('triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 52480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 5
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + 52480*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp3 / tmp5
        tmp7 = 0.0
        tmp8 = tl.where(tmp1, tmp6, tmp7)
        tmp9 = tmp0 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/fx/cfx67jhu5rna7gnffim4jrqmbm2dyojfnddcx55muns4nb2cys27.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_3
# Graph fragment:
#   %div_10 : [num_users=9] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_45), kwargs = {})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_57, %div_10, %full_default_3), kwargs = {})
#   %mul_251 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_21), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_251, [1], True), kwargs = {})
triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5 = async_compile.triton('triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 5
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 5*x0), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/i6/ci6gcd7crb6eouqz3atfqlxy42jl5i5cgnyw4bcxln2bqd6yrn76.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_29, _shift_logits_30, cross_entropy_loss_7], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
# Source node to ATen node mapping:
#   _shift_logits_29 => div_7
#   _shift_logits_30 => tanh_7
#   cross_entropy_loss => full_default_3
#   cross_entropy_loss_7 => sub_119
# Graph fragment:
#   %div_10 : [num_users=9] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_45), kwargs = {})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_21 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_57, %div_10, %full_default_3), kwargs = {})
#   %mul_251 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_21), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_7, 30.0), kwargs = {})
#   %tanh_7 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_7,), kwargs = {})
#   %convert_element_type_default_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tanh_7, torch.float32), kwargs = {})
#   %mul_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, 1), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_2, %amax_default_1), kwargs = {})
#   %mul_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_1, 30.0), kwargs = {})
#   %sub_119 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_3, %log_7), kwargs = {})
#   %exp_10 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_119,), kwargs = {})
#   %mul_252 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_10, %sum_29), kwargs = {})
#   %sub_135 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_251, %mul_252), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_135, torch.bfloat16), kwargs = {})
#   %mul_253 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_53, 30.0), kwargs = {})
#   %convert_element_type_54 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_253, torch.float32), kwargs = {})
#   %mul_254 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_1, %convert_element_type_default_1), kwargs = {})
#   %sub_136 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_254), kwargs = {})
#   %mul_255 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_54, %sub_136), kwargs = {})
#   %convert_element_type_56 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_255, torch.bfloat16), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_56, 30.0), kwargs = {})
triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6 = async_compile.triton('triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 262400
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr3 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp10 = tl.load(in_ptr4 + (x2), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tmp3 / tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp1, tmp6, tmp7)
    tmp9 = tmp0 * tmp8
    tmp11 = 0.03333333333333333
    tmp12 = tmp10 * tmp11
    tmp13 = libdevice.tanh(tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp18 = tmp16 - tmp17
    tmp19 = 30.0
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 - tmp21
    tmp23 = tl_math.exp(tmp22)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp9 - tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp19
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp14 * tmp14
    tmp31 = tmp15 - tmp30
    tmp32 = tmp29 * tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp11
    tl.store(out_ptr1 + (x2), tmp34, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ht/chtn7dvfgfzi5lochx4dhzsipmzisah4vad2dtqvd62cxo57incv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_108 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_17, torch.float32), kwargs = {})
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_6, primals_2, sym_size_int_8, mm, amax_default_8, log, mm_1, amax_default_7, log_1, mm_2, amax_default_6, log_2, mm_3, amax_default_5, log_3, mm_4, amax_default_4, log_4, mm_5, amax_default_3, log_5, mm_6, amax_default_2, log_6, mm_7, amax_default_1, log_7, mm_8, amax_default, log_8, convert_element_type_45, ne_55, where_18, permute_10, ne_57, where_20, ne_59, where_22, ne_61, where_24, ne_63, where_26, ne_65, where_28, ne_67, where_30, ne_69, where_32, ne_71, where_34, tangents_1 = args
    args.clear()
    s3 = primals_6
    s0 = primals_2
    assert_size_stride(mm, ((8 + s3) // 9, 262400), (262400, 1))
    assert_size_stride(amax_default_8, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(log, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(mm_1, ((8 + s3) // 9, 262400), (262400, 1))
    assert_size_stride(amax_default_7, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(log_1, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(mm_2, ((8 + s3) // 9, 262400), (262400, 1))
    assert_size_stride(amax_default_6, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(log_2, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(mm_3, ((8 + s3) // 9, 262400), (262400, 1))
    assert_size_stride(amax_default_5, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(log_3, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(mm_4, ((8 + s3) // 9, 262400), (262400, 1))
    assert_size_stride(amax_default_4, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(log_4, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(mm_5, ((8 + s3) // 9, 262400), (262400, 1))
    assert_size_stride(amax_default_3, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(log_5, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(mm_6, ((8 + s3) // 9, 262400), (262400, 1))
    assert_size_stride(amax_default_2, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(log_6, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(mm_7, ((8 + s3) // 9, 262400), (262400, 1))
    assert_size_stride(amax_default_1, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(log_7, ((8 + s3) // 9, 1), (1, 1))
    assert_size_stride(mm_8, (s3 + ((-8)*((8 + s3) // 9)), 262400), (262400, 1))
    assert_size_stride(amax_default, (s3 + ((-8)*((8 + s3) // 9)), 1), (1, 1))
    assert_size_stride(log_8, (s3 + ((-8)*((8 + s3) // 9)), 1), (1, 1))
    assert_size_stride(convert_element_type_45, (), ())
    assert_size_stride(ne_55, (s0 + ((-8)*((8 + s0) // 9)), 1), (1, 1))
    assert_size_stride(where_18, (s0 + ((-8)*((8 + s0) // 9)), 1), (1, 1))
    assert_size_stride(permute_10, (262400, 2048), (2048, 1))
    assert_size_stride(ne_57, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(where_20, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(ne_59, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(where_22, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(ne_61, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(where_24, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(ne_63, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(where_26, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(ne_65, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(where_28, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(ne_67, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(where_30, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(ne_69, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(where_32, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(ne_71, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(where_34, ((8 + s0) // 9, 1), (1, 1))
    assert_size_stride(tangents_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((s3 + ((-8)*((8 + s3) // 9)), 1), (1, s3 + ((-8)*((8 + s3) // 9))), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_0_xnumel = s3 + ((-8)*((8 + s3) // 9))
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_0.run(where_18, ne_55, tangents_1, convert_element_type_45, buf0, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_0_xnumel, 262400, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_0_xnumel), stream=stream0)
        buf1 = empty_strided_cuda((s3 + ((-8)*((8 + s3) // 9)), 262400), (262400, 1), torch.bfloat16)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_33, _shift_logits_34, cross_entropy_loss_8], Original ATen: [aten.div, aten.nll_loss_backward, aten.nll_loss_forward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_1_xnumel = ((-2099200)*((8 + s3) // 9)) + 262400*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_1.run(buf2, where_18, ne_55, tangents_1, convert_element_type_45, mm_8, amax_default, log_8, buf0, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_1_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_1_xnumel), stream=stream0)
        del amax_default
        del buf0
        del log_8
        del mm_8
        del ne_55
        del where_18
        buf3 = empty_strided_cuda((s3 + ((-8)*((8 + s3) // 9)), 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_33, _shift_logits_34], Original ATen: [aten.div, aten.tanh, aten.mul, aten.tanh_backward, aten.mm]
        extern_kernels.mm(buf2, permute_10, out=buf3)
        del buf2
        buf4 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(buf4, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_3_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_3.run(where_20, buf4, triton_poi_fused_nll_loss_backward_3_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_3_xnumel), stream=stream0)
        del where_20
        buf6 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf4, ne_57, tangents_1, convert_element_type_45, buf6, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf7 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5.run(buf6, buf7, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel, 5, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        buf9 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_29, _shift_logits_30, cross_entropy_loss_7], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6.run(buf4, ne_57, tangents_1, convert_element_type_45, mm_7, amax_default_1, log_7, buf7, buf9, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel), stream=stream0)
        del amax_default_1
        del log_7
        del mm_7
        del ne_57
        buf10 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.tanh_backward, aten.div, aten.mm]
        extern_kernels.mm(buf9, permute_10, out=buf10)
        buf11 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(buf11, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_3_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_3.run(where_22, buf11, triton_poi_fused_nll_loss_backward_3_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_3_xnumel), stream=stream0)
        del where_22
        buf13 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf11, ne_59, tangents_1, convert_element_type_45, buf13, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5.run(buf13, buf14, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel, 5, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        buf16 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_25, _shift_logits_26, cross_entropy_loss_6], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6.run(buf11, ne_59, tangents_1, convert_element_type_45, mm_6, amax_default_2, log_6, buf14, buf16, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel), stream=stream0)
        del amax_default_2
        del log_6
        del mm_6
        del ne_59
        buf17 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.tanh_backward, aten.div, aten.mm]
        extern_kernels.mm(buf16, permute_10, out=buf17)
        buf18 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(buf18, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_3_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_3.run(where_24, buf18, triton_poi_fused_nll_loss_backward_3_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_3_xnumel), stream=stream0)
        del where_24
        buf20 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf18, ne_61, tangents_1, convert_element_type_45, buf20, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf21 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5.run(buf20, buf21, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel, 5, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        buf23 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_21, _shift_logits_22, cross_entropy_loss_5], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6.run(buf18, ne_61, tangents_1, convert_element_type_45, mm_5, amax_default_3, log_5, buf21, buf23, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel), stream=stream0)
        del amax_default_3
        del log_5
        del mm_5
        del ne_61
        buf24 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.tanh_backward, aten.div, aten.mm]
        extern_kernels.mm(buf23, permute_10, out=buf24)
        buf25 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(buf25, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_3_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_3.run(where_26, buf25, triton_poi_fused_nll_loss_backward_3_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_3_xnumel), stream=stream0)
        del where_26
        buf27 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf25, ne_63, tangents_1, convert_element_type_45, buf27, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf28 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5.run(buf27, buf28, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel, 5, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        buf30 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_17, _shift_logits_18, cross_entropy_loss_4], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6.run(buf25, ne_63, tangents_1, convert_element_type_45, mm_4, amax_default_4, log_4, buf28, buf30, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel), stream=stream0)
        del amax_default_4
        del log_4
        del mm_4
        del ne_63
        buf31 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.tanh_backward, aten.div, aten.mm]
        extern_kernels.mm(buf30, permute_10, out=buf31)
        buf32 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(buf32, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_3_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_3.run(where_28, buf32, triton_poi_fused_nll_loss_backward_3_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_3_xnumel), stream=stream0)
        del where_28
        buf34 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf32, ne_65, tangents_1, convert_element_type_45, buf34, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf35 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5.run(buf34, buf35, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel, 5, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        buf37 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_13, _shift_logits_14, cross_entropy_loss_3], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6.run(buf32, ne_65, tangents_1, convert_element_type_45, mm_3, amax_default_5, log_3, buf35, buf37, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel), stream=stream0)
        del amax_default_5
        del log_3
        del mm_3
        del ne_65
        buf38 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.tanh_backward, aten.div, aten.mm]
        extern_kernels.mm(buf37, permute_10, out=buf38)
        buf39 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(buf39, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_3_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_3.run(where_30, buf39, triton_poi_fused_nll_loss_backward_3_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_3_xnumel), stream=stream0)
        del where_30
        buf41 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf39, ne_67, tangents_1, convert_element_type_45, buf41, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf42 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5.run(buf41, buf42, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel, 5, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        buf44 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_9, _shift_logits_10, cross_entropy_loss_2], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6.run(buf39, ne_67, tangents_1, convert_element_type_45, mm_2, amax_default_6, log_2, buf42, buf44, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel), stream=stream0)
        del amax_default_6
        del log_2
        del mm_2
        del ne_67
        buf45 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.tanh_backward, aten.div, aten.mm]
        extern_kernels.mm(buf44, permute_10, out=buf45)
        buf46 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(buf46, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_3_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_3.run(where_32, buf46, triton_poi_fused_nll_loss_backward_3_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_3_xnumel), stream=stream0)
        del where_32
        buf48 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf46, ne_69, tangents_1, convert_element_type_45, buf48, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf49 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5.run(buf48, buf49, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel, 5, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        buf51 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_5, _shift_logits_6, cross_entropy_loss_1], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6.run(buf46, ne_69, tangents_1, convert_element_type_45, mm_1, amax_default_7, log_1, buf49, buf51, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel), stream=stream0)
        del amax_default_7
        del log_1
        del mm_1
        del ne_69
        buf52 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.tanh_backward, aten.div, aten.mm]
        extern_kernels.mm(buf51, permute_10, out=buf52)
        buf53 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_2_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_2.run(buf53, triton_poi_fused_nll_loss_backward_2_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_2_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_3_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_3.run(where_34, buf53, triton_poi_fused_nll_loss_backward_3_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_3_xnumel), stream=stream0)
        del where_34
        buf55 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4.run(buf53, ne_71, tangents_1, convert_element_type_45, buf55, triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_4_xnumel), stream=stream0)
        buf56 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten._log_softmax_backward_data]
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5.run(buf55, buf56, triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel, 5, grid=grid(triton_per_fused__log_softmax_backward_data_div_nll_loss_backward_nll_loss_forward_5_xnumel), stream=stream0)
        del buf55
        buf58 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, _shift_logits_1, _shift_logits_2], Original ATen: [aten.div, aten.nll_loss_forward, aten.nll_loss_backward, aten.tanh, aten._log_softmax, aten._log_softmax_backward_data, aten._to_copy, aten.mul, aten.tanh_backward]
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel = 262400*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6.run(buf53, ne_71, tangents_1, convert_element_type_45, mm, amax_default_8, log, buf56, buf58, triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel, grid=grid(triton_poi_fused__log_softmax__log_softmax_backward_data__to_copy_div_mul_nll_loss_backward_nll_loss_forward_tanh_tanh_backward_6_xnumel), stream=stream0)
        del amax_default_8
        del buf53
        del buf56
        del convert_element_type_45
        del log
        del mm
        del ne_71
        del tangents_1
        buf59 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.tanh_backward, aten.div, aten.mm]
        extern_kernels.mm(buf58, permute_10, out=buf59)
        del buf58
        del permute_10
        buf69 = empty_strided_cuda((s3, 2048), (2048, 1), torch.float32)
        buf60 = reinterpret_tensor(buf69, ((8 + s3) // 9, 2048), (2048, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf59, buf60, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf59
        buf61 = reinterpret_tensor(buf69, ((8 + s3) // 9, 2048), (2048, 1), 2048*((8 + s3) // 9))  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf52, buf61, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf52
        buf62 = reinterpret_tensor(buf69, ((8 + s3) // 9, 2048), (2048, 1), 4096*((8 + s3) // 9))  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf45, buf62, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf45
        buf63 = reinterpret_tensor(buf69, ((8 + s3) // 9, 2048), (2048, 1), 6144*((8 + s3) // 9))  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf38, buf63, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf38
        buf64 = reinterpret_tensor(buf69, ((8 + s3) // 9, 2048), (2048, 1), 8192*((8 + s3) // 9))  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf31, buf64, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf31
        buf65 = reinterpret_tensor(buf69, ((8 + s3) // 9, 2048), (2048, 1), 10240*((8 + s3) // 9))  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf24, buf65, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf24
        buf66 = reinterpret_tensor(buf69, ((8 + s3) // 9, 2048), (2048, 1), 12288*((8 + s3) // 9))  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf17, buf66, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf17
        buf67 = reinterpret_tensor(buf69, ((8 + s3) // 9, 2048), (2048, 1), 14336*((8 + s3) // 9))  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf10, buf67, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf10
        buf68 = reinterpret_tensor(buf69, (s3 + ((-8)*((8 + s3) // 9)), 2048), (2048, 1), 16384*((8 + s3) // 9))  # alias
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = ((-16384)*((8 + s3) // 9)) + 2048*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(buf3, buf68, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        del buf3
    return (None, None, None, None, None, None, reinterpret_tensor(buf69, (1, s3, 2048), (2048*s3, 2048, 1), 0), None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_6 = 533
    primals_2 = 533
    sym_size_int_8 = 53
    mm = rand_strided((60, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_8 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_1 = rand_strided((60, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_7 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_1 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_2 = rand_strided((60, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_6 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_2 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_3 = rand_strided((60, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_5 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_3 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_4 = rand_strided((60, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_4 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_4 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_5 = rand_strided((60, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_3 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_5 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_6 = rand_strided((60, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_2 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_6 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_7 = rand_strided((60, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default_1 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_7 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    mm_8 = rand_strided((53, 262400), (262400, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_default = rand_strided((53, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log_8 = rand_strided((53, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_45 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    ne_55 = rand_strided((53, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_18 = rand_strided((53, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    permute_10 = rand_strided((262400, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    ne_57 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_20 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_59 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_22 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_61 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_24 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_63 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_26 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_65 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_28 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_67 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_30 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_69 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_32 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    ne_71 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.bool)
    where_34 = rand_strided((60, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_6, primals_2, sym_size_int_8, mm, amax_default_8, log, mm_1, amax_default_7, log_1, mm_2, amax_default_6, log_2, mm_3, amax_default_5, log_3, mm_4, amax_default_4, log_4, mm_5, amax_default_3, log_5, mm_6, amax_default_2, log_6, mm_7, amax_default_1, log_7, mm_8, amax_default, log_8, convert_element_type_45, ne_55, where_18, permute_10, ne_57, where_20, ne_59, where_22, ne_61, where_24, ne_63, where_26, ne_65, where_28, ne_67, where_30, ne_69, where_32, ne_71, where_34, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
