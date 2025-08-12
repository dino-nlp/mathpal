# AOT ID: ['59_forward']
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


# kernel path: /tmp/torchinductor_root/tq/ctqu7o5qynpq2hu6iyr3tw5rcrzat2u2oveucianb5zi7ztjlyoz.py
# Topologically Sorted Source Nodes: [_shift_logits], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/mm/cmmcbjzq3rt3r3r2zn55bjyxwu3oxjwhn3633nai2qjitymjvg5j.py
# Topologically Sorted Source Nodes: [_shift_logits_4], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits_4 => convert_element_type_5
# Graph fragment:
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*((8 + ks0) // 9)), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/my/cmypsgly4qkyondod37dhatofftrmzznhp6htk7gmdnsht7cx6rx.py
# Topologically Sorted Source Nodes: [_shift_logits_8], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits_8 => convert_element_type_10
# Graph fragment:
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096*((8 + ks0) // 9)), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/so/csonpdzsqpoh475xhtsgfmkkhrpzas5apqxo52utyivus7rlvpzd.py
# Topologically Sorted Source Nodes: [_shift_logits_12], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits_12 => convert_element_type_15
# Graph fragment:
#   %convert_element_type_15 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_3, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 6144*((8 + ks0) // 9)), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/h7/ch7bzeemdfs4hdtbplqe5wjbr3ekyk2ogubndkx3xlekz5sohdpy.py
# Topologically Sorted Source Nodes: [_shift_logits_16], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits_16 => convert_element_type_20
# Graph fragment:
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_4, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192*((8 + ks0) // 9)), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/de/cdew6fc2jfnuqhi6spdm6qiwpbmaxu7paq2fnbq7wuuq2z35q63n.py
# Topologically Sorted Source Nodes: [_shift_logits_20], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits_20 => convert_element_type_25
# Graph fragment:
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_5, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 10240*((8 + ks0) // 9)), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/bp/cbpglm34dovmdbggqfow7ghwlpiasa2t4bwwosvkjvmderdgvrzf.py
# Topologically Sorted Source Nodes: [_shift_logits_24], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits_24 => convert_element_type_30
# Graph fragment:
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_6, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_6 = async_compile.triton('triton_poi_fused__to_copy_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 12288*((8 + ks0) // 9)), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/7l/c7ltptdevmhek6qjxtkpzlzsansylbimr5m5yh2twp3yw4kcjid3.py
# Topologically Sorted Source Nodes: [_shift_logits_28], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits_28 => convert_element_type_35
# Graph fragment:
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_7, torch.bfloat16), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 14336*((8 + ks0) // 9)), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/hg/chgc4fix4bjm5bnm7zgk3p3wvd7pkj7z3cmzpvqqapb4ocf2hhvc.py
# Topologically Sorted Source Nodes: [_shift_logits_32], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   _shift_logits_32 => convert_element_type_40
# Graph fragment:
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_8, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384*((8 + ks0) // 9)), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/gh/cgheroikqwy77g4jymurdfhlfvbxxiash7r3g644mt5pil34lipu.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %slice_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%permute, %slice_1, 1, 0, -1), kwargs = {})
triton_poi_fused_9 = async_compile.triton('triton_poi_fused_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*i64', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp4 = tl.load(in_ptr1 + (x0), xmask)
    tmp0 = x0
    tmp1 = (-1) + ks0
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (1 + x0), tmp2 & xmask, other=0.0)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/nk/cnkqj3cy3zsqh3v4mbrnkoo76zlw5fupt7vxuk7z5iukgwadsah4.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
# Source node to ATen node mapping:
#   setitem_1 => full_default, index_put
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], -100), kwargs = {dtype: torch.int64, layout: torch.strided, device: cpu, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%slice_6, [%eq_14], %full_default), kwargs = {})
triton_poi_fused_index_put_lift_fresh_10 = async_compile.triton('triton_poi_fused_index_put_lift_fresh_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr1': '*i64', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_lift_fresh_10', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_put_lift_fresh_10(in_ptr0, in_ptr1, in_ptr2, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0), xmask)
    tmp7 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = (-1) + ks0
    tmp5 = tmp3 < tmp4
    tmp6 = tl.load(in_ptr1 + (1 + x0), tmp5 & xmask, other=0.0)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.full([1], -100, tl.int64)
    tmp10 = tl.where(tmp2, tmp9, tmp8)
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/n5/cn5atqrui3d244gqxzmcvddgqgg4ls53d3mcijhfja36ywfhyemb.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_69 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_16, -100), kwargs = {})
#   %where_32 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_69, %unsqueeze_16, %full_default_2), kwargs = {})
triton_poi_fused_nll_loss_backward_nll_loss_forward_11 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i1', 'out_ptr1': '*i64', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_11(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_ptr0 + (x0 + ((8 + ks0) // 9)), xmask)
    tmp0 = x0 + ((8 + ks0) // 9)
    tmp1 = (-1) + ks0
    tmp2 = tmp0 == tmp1
    tmp3 = tmp0 < tmp1
    tmp4 = tl.load(in_ptr0 + (x0 + ((8 + ks0) // 9)), tmp3 & xmask, other=0.0)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tl.full([1], -100, tl.int64)
    tmp8 = tl.where(tmp2, tmp7, tmp6)
    tmp9 = tmp8 != tmp7
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tl.where(tmp9, tmp8, tmp10)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ni/cnihhyeoo6x7ezcrwfxyxnwrozs2yt5keu37ztsdth3jhjlholol.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_71 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_17, -100), kwargs = {})
#   %where_34 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_71, %unsqueeze_17, %full_default_2), kwargs = {})
triton_poi_fused_nll_loss_backward_nll_loss_forward_12 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i1', 'out_ptr1': '*i64', 'ks0': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_12(in_ptr0, out_ptr0, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_ptr0 + (x0), xmask)
    tmp0 = x0
    tmp1 = (-1) + ks0
    tmp2 = tmp0 == tmp1
    tmp3 = tmp0 < tmp1
    tmp4 = tl.load(in_ptr0 + (x0), tmp3 & xmask, other=0.0)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tl.full([1], -100, tl.int64)
    tmp8 = tl.where(tmp2, tmp7, tmp6)
    tmp9 = tmp8 != tmp7
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tl.where(tmp9, tmp8, tmp10)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tl.store(out_ptr1 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/at/cate3xii25izazqr4jvvpulhouwv6ig7q6wzdzm5mdxvgr4kavsj.py
# Topologically Sorted Source Nodes: [_shift_logits_1, _shift_logits_2], Original ATen: [aten.div, aten.tanh]
# Source node to ATen node mapping:
#   _shift_logits_1 => div
#   _shift_logits_2 => tanh
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 30.0), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %convert_element_type_default_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tanh, torch.float32), kwargs = {})
#   %mul_tensor_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, 1), kwargs = {})
#   %amax_default_8 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_16, [1], True), kwargs = {})
triton_red_fused_div_tanh_13 = async_compile.triton('triton_red_fused_div_tanh_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_div_tanh_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_div_tanh_13(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 52480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 52480*x0), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.03333333333333333
        tmp2 = tmp0 * tmp1
        tmp3 = libdevice.tanh(tmp2)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/2e/c2ewnp3vdpo3z4hq4uijsvo6iksxf4qio4zimwdrimb747uwgqm2.py
# Topologically Sorted Source Nodes: [_shift_logits_1, _shift_logits_2], Original ATen: [aten.div, aten.tanh]
# Source node to ATen node mapping:
#   _shift_logits_1 => div
#   _shift_logits_2 => tanh
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 30.0), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %convert_element_type_default_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tanh, torch.float32), kwargs = {})
#   %mul_tensor_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, 1), kwargs = {})
#   %amax_default_8 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_16, [1], True), kwargs = {})
triton_per_fused_div_tanh_14 = async_compile.triton('triton_per_fused_div_tanh_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_tanh_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_tanh_14(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/b7/cb7v57lyihwzdk5iutkiabu653pamolfmhu7hs3nvk67nio4l75b.py
# Topologically Sorted Source Nodes: [_shift_logits_1, _shift_logits_2, cross_entropy_loss], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
# Source node to ATen node mapping:
#   _shift_logits_1 => div
#   _shift_logits_2 => tanh
#   cross_entropy_loss => exp, sum_1
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 30.0), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %convert_element_type_default_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tanh, torch.float32), kwargs = {})
#   %mul_tensor_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, 1), kwargs = {})
#   %sub_tensor_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_16, %amax_default_8), kwargs = {})
#   %mul_tensor_17 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_8, 30.0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_17,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
triton_red_fused__log_softmax_div_tanh_15 = async_compile.triton('triton_red_fused__log_softmax_div_tanh_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_div_tanh_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_div_tanh_15(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 52480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x1 = xindex // 5
    tmp7 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + 52480*x3), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = 0.03333333333333333
        tmp2 = tmp0 * tmp1
        tmp3 = libdevice.tanh(tmp2)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 1.0
        tmp6 = tmp4 * tmp5
        tmp8 = tmp6 - tmp7
        tmp9 = 30.0
        tmp10 = tmp8 * tmp9
        tmp11 = tl_math.exp(tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ey/ceylblttjssybx4lrbnhn3wcquvxiwzjvigreldkczbh6ne3fysp.py
# Topologically Sorted Source Nodes: [_shift_logits_1, _shift_logits_2, cross_entropy_loss], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
# Source node to ATen node mapping:
#   _shift_logits_1 => div
#   _shift_logits_2 => tanh
#   cross_entropy_loss => exp, log, sum_1
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 30.0), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %convert_element_type_default_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tanh, torch.float32), kwargs = {})
#   %mul_tensor_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_default_8, 1), kwargs = {})
#   %sub_tensor_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_16, %amax_default_8), kwargs = {})
#   %mul_tensor_17 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_8, 30.0), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_17,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %log : [num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
triton_per_fused__log_softmax_div_tanh_16 = async_compile.triton('triton_per_fused__log_softmax_div_tanh_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_div_tanh_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_div_tanh_16(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl_math.log(tmp4)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/bb/cbbt5r5c5v3npqq5cr32ayc2u37hbpc7cwczeae7nnyzgjwuyvwb.py
# Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_3, ne_4, neg, sum_3, where_1
# Graph fragment:
#   %ne_4 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_18, -100), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %neg, %full_default_3), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_1,), kwargs = {})
triton_red_fused_nll_loss_forward_17 = async_compile.triton('triton_red_fused_nll_loss_forward_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_forward_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ry/cryqgjuecqoenpjwml4ywc7qxlgeaqr3oilrpj7pagyy6rs3zbyz.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_1], Original ATen: [aten.nll_loss_forward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_3
#   cross_entropy_loss_1 => ne_10, neg_1, sum_6, where_3
# Graph fragment:
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_10 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_28, -100), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_1,), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_10, %neg_1, %full_default_3), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_3,), kwargs = {})
triton_red_fused_nll_loss_forward_18 = async_compile.triton('triton_red_fused_nll_loss_forward_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_forward_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0 + ((8 + ks0) // 9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + ((8 + ks0) // 9)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + ((8 + ks0) // 9), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/y2/cy2hpuok5qwupsi3ztkemsweywkhpj6ec7yqgou3qjmq3nlas67q.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_2], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_2 => ne_16, neg_2, sum_9, where_5
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_16 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_38, -100), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_2,), kwargs = {})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_16, %neg_2, %full_default_3), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_5,), kwargs = {})
#   %ne_67 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_15, -100), kwargs = {})
#   %where_30 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_67, %unsqueeze_15, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_19 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0 + 2*((8 + ks0) // 9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 2*((8 + ks0) // 9)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + 2*((8 + ks0) // 9), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/tc/ctcihknyacjfh53zkg7izqjgxu3xi56jexdv7rjip2veymdqemgf.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_3 => ne_22, neg_3, sum_12, where_7
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_22 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_48, -100), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_3,), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_22, %neg_3, %full_default_3), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_7,), kwargs = {})
#   %ne_65 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_14, -100), kwargs = {})
#   %where_28 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_65, %unsqueeze_14, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_20 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0 + 3*((8 + ks0) // 9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 3*((8 + ks0) // 9)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + 3*((8 + ks0) // 9), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/dv/cdvu2b7jrw2ht5wrzhfapospjbk72f3ez4yjhtnhibm2avmvuphp.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_4], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_4 => ne_28, neg_4, sum_15, where_9
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_28 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_58, -100), kwargs = {})
#   %neg_4 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_4,), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_28, %neg_4, %full_default_3), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_9,), kwargs = {})
#   %ne_63 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_13, -100), kwargs = {})
#   %where_26 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_63, %unsqueeze_13, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_21 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0 + 4*((8 + ks0) // 9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 4*((8 + ks0) // 9)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + 4*((8 + ks0) // 9), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/fz/cfz4drcu2643mccqego2ip427lm4q3hscph2nadjtlhi5dl62lmc.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_5], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_5 => ne_34, neg_5, sum_18, where_11
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_34 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_68, -100), kwargs = {})
#   %neg_5 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_5,), kwargs = {})
#   %where_11 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_34, %neg_5, %full_default_3), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_11,), kwargs = {})
#   %ne_61 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_12, -100), kwargs = {})
#   %where_24 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_61, %unsqueeze_12, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_22 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0 + 5*((8 + ks0) // 9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 5*((8 + ks0) // 9)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + 5*((8 + ks0) // 9), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/nx/cnxy6mtfkkmuiyo5rmhlj5wd4yedyt23fk6cg2unmrt5q6ix5mtb.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_6], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_6 => ne_40, neg_6, sum_21, where_13
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_40 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_78, -100), kwargs = {})
#   %neg_6 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_6,), kwargs = {})
#   %where_13 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_40, %neg_6, %full_default_3), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_13,), kwargs = {})
#   %ne_59 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_11, -100), kwargs = {})
#   %where_22 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_59, %unsqueeze_11, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_23 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0 + 6*((8 + ks0) // 9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 6*((8 + ks0) // 9)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + 6*((8 + ks0) // 9), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ni/cni2t2qaip35i4ysu2apbjthdwmakuhz4oscsduwqafpcieb6zeg.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_7], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_7 => ne_46, neg_7, sum_24, where_15
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_46 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_88, -100), kwargs = {})
#   %neg_7 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_7,), kwargs = {})
#   %where_15 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_46, %neg_7, %full_default_3), kwargs = {})
#   %sum_24 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_15,), kwargs = {})
#   %ne_57 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_10, -100), kwargs = {})
#   %where_20 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_57, %unsqueeze_10, %full_default_2), kwargs = {})
triton_red_fused_nll_loss_backward_nll_loss_forward_24 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': (8,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0 + 7*((8 + ks0) // 9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 7*((8 + ks0) // 9)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + 7*((8 + ks0) // 9), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp34, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/pd/cpdisacqd364bwfxcdjlvmte6auurk32fswlsufssnrqqqjpieph.py
# Topologically Sorted Source Nodes: [cross_entropy_loss, loss, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, cross_entropy_loss_8, loss_8, tensor, loss_9], Original ATen: [aten.nll_loss_forward, aten.add, aten._to_copy, aten.div, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy_loss => full_default_2, full_default_3
#   cross_entropy_loss_8 => ne_52, neg_8, sum_27, where_17
#   loss => add_105
#   loss_1 => add_133
#   loss_2 => add_161
#   loss_3 => add_189
#   loss_4 => add_217
#   loss_5 => add_245
#   loss_6 => add_273
#   loss_7 => add_301
#   loss_8 => add_329
#   loss_9 => div_9
#   tensor => convert_element_type_45
# Graph fragment:
#   %full_default_2 : [num_users=18] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=9] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, 0.0), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_105, %sum_6), kwargs = {})
#   %add_161 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %sum_9), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_161, %sum_12), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_189, %sum_15), kwargs = {})
#   %add_245 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_217, %sum_18), kwargs = {})
#   %add_273 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_245, %sum_21), kwargs = {})
#   %add_301 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_273, %sum_24), kwargs = {})
#   %ne_52 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%getitem_98, -100), kwargs = {})
#   %neg_8 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze_8,), kwargs = {})
#   %where_17 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_52, %neg_8, %full_default_3), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_17,), kwargs = {})
#   %add_329 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_301, %sum_27), kwargs = {})
#   %convert_element_type_45 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.float32), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_329, %convert_element_type_45), kwargs = {})
#   %ne_55 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_9, -100), kwargs = {})
#   %where_18 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_55, %unsqueeze_9, %full_default_2), kwargs = {})
triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_25 = async_compile.triton('triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*i64', 'out_ptr3': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': (17,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 1, 'backend_hash': '04DE1D738B8CBCE77E01EA090596738070D0713A8B4EB68F51DA82EC6DDBC324', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr1, out_ptr2, out_ptr3, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp34 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp5 = tl.load(in_ptr0 + (r0 + 8*((8 + ks0) // 9)), rmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp0 = r0 + 8*((8 + ks0) // 9)
        tmp1 = (-1) + ks0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp0 < tmp1
        tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r0 + 8*((8 + ks0) // 9), [XBLOCK, RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tl.full([1, 1], -100, tl.int64)
        tmp8 = tl.where(tmp2, tmp7, tmp6)
        tmp9 = tmp8 != tmp7
        tmp10 = tl.full([1, 1], 0, tl.int64)
        tmp11 = tl.where(tmp9, tmp8, tmp10)
        tmp12 = tl.full([XBLOCK, RBLOCK], 262400, tl.int32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp11 < 0
        tmp15 = tl.where(tmp14, tmp13, tmp11)
        tl.device_assert(((0 <= tmp15) & (tmp15 < 262400)) | ~(rmask), "index out of bounds: 0 <= tmp15 < 262400")
        tmp17 = tl.load(in_ptr1 + (tmp15 + 262400*r0), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = 0.03333333333333333
        tmp19 = tmp17 * tmp18
        tmp20 = libdevice.tanh(tmp19)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 1.0
        tmp23 = tmp21 * tmp22
        tmp25 = tmp23 - tmp24
        tmp26 = 30.0
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 - tmp28
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp9, tmp30, tmp31)
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
        tmp35 = _tmp34 + tmp33
        _tmp34 = tl.where(rmask, tmp35, _tmp34)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp11, rmask)
    tmp34 = tl.sum(_tmp34, 1)[:, None]
    tmp36 = tl.load(in_ptr4 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, 1])
    tmp39 = tl.load(in_out_ptr0 + (0))
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, 1])
    tmp43 = tl.load(in_ptr5 + (0))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK, 1])
    tmp46 = tl.load(in_ptr6 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, 1])
    tmp49 = tl.load(in_ptr7 + (0))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, 1])
    tmp52 = tl.load(in_ptr8 + (0))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK, 1])
    tmp55 = tl.load(in_ptr9 + (0))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, 1])
    tmp58 = tl.load(in_ptr10 + (0))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK, 1])
    tmp61 = tl.load(in_ptr11 + (0))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK, 1])
    tmp38 = tmp37.to(tl.float32)
    tmp41 = 0.0
    tmp42 = tmp40 + tmp41
    tmp45 = tmp42 + tmp44
    tmp48 = tmp45 + tmp47
    tmp51 = tmp48 + tmp50
    tmp54 = tmp51 + tmp53
    tmp57 = tmp54 + tmp56
    tmp60 = tmp57 + tmp59
    tmp63 = tmp60 + tmp62
    tmp64 = tmp63 + tmp34
    tmp65 = tmp64 / tmp38
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp38, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp65, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    s0 = primals_2
    s1 = primals_4
    s3 = primals_6
    assert_size_stride(primals_1, (262400, 2048), (2048, 1))
    assert_size_stride(primals_3, (1, s0), (s0, 1))
    assert_size_stride(primals_5, (1, s0), (s0, 1))
    assert_size_stride(primals_7, (1, s3, 2048), (2048*s3, 2048, 1))
    assert_size_stride(primals_8, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, s0), (s0, 1), torch.int64)
        buf4 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_0_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_7, buf4, triton_poi_fused__to_copy_0_xnumel, grid=grid(triton_poi_fused__to_copy_0_xnumel), stream=stream0)
        buf12 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_4], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_1_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_7, buf12, s3, triton_poi_fused__to_copy_1_xnumel, grid=grid(triton_poi_fused__to_copy_1_xnumel), stream=stream0)
        buf20 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_8], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_2_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_7, buf20, s3, triton_poi_fused__to_copy_2_xnumel, grid=grid(triton_poi_fused__to_copy_2_xnumel), stream=stream0)
        buf28 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_12], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_3.run(primals_7, buf28, s3, triton_poi_fused__to_copy_3_xnumel, grid=grid(triton_poi_fused__to_copy_3_xnumel), stream=stream0)
        buf36 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_16], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(primals_7, buf36, s3, triton_poi_fused__to_copy_4_xnumel, grid=grid(triton_poi_fused__to_copy_4_xnumel), stream=stream0)
        buf44 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_20], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_5_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(primals_7, buf44, s3, triton_poi_fused__to_copy_5_xnumel, grid=grid(triton_poi_fused__to_copy_5_xnumel), stream=stream0)
        buf52 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_24], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_6_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_6.run(primals_7, buf52, s3, triton_poi_fused__to_copy_6_xnumel, grid=grid(triton_poi_fused__to_copy_6_xnumel), stream=stream0)
        buf60 = empty_strided_cuda(((8 + s3) // 9, 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_28], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7_xnumel = 2048*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(primals_7, buf60, s3, triton_poi_fused__to_copy_7_xnumel, grid=grid(triton_poi_fused__to_copy_7_xnumel), stream=stream0)
        buf68 = empty_strided_cuda((s3 + ((-8)*((8 + s3) // 9)), 2048), (2048, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_32], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_8_xnumel = ((-16384)*((8 + s3) // 9)) + 2048*s3
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_7, buf68, s3, triton_poi_fused__to_copy_8_xnumel, grid=grid(triton_poi_fused__to_copy_8_xnumel), stream=stream0)
        del primals_7
        buf1 = empty_strided_cuda((1, s0), (s0, 1), torch.int64)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(primals_3, buf0, buf1, s0, s0, grid=grid(s0), stream=stream0)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh, aten.index_put]
        triton_poi_fused_index_put_lift_fresh_10_xnumel = (-1) + s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_put_lift_fresh_10.run(primals_5, primals_3, buf0, buf1, s0, triton_poi_fused_index_put_lift_fresh_10_xnumel, grid=grid(triton_poi_fused_index_put_lift_fresh_10_xnumel), stream=stream0)
        del buf0
        del primals_3
        del primals_5
        buf5 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf5)
        del buf4
        buf13 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_4], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf12, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf13)
        del buf12
        buf21 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_8], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf20, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf21)
        del buf20
        buf29 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_12], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf28, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf29)
        del buf28
        buf37 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_16], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf36, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf37)
        del buf36
        buf45 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_20], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf44, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf45)
        del buf44
        buf53 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_24], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf53)
        del buf52
        buf61 = empty_strided_cuda(((8 + s3) // 9, 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_28], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf60, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf61)
        del buf60
        buf69 = empty_strided_cuda((s3 + ((-8)*((8 + s3) // 9)), 262400), (262400, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [_shift_logits_32], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf68, reinterpret_tensor(primals_1, (2048, 262400), (1, 2048), 0), out=buf69)
        del buf68
        buf91 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.bool)
        buf92 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_11_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_11.run(buf1, buf91, buf92, s0, triton_poi_fused_nll_loss_backward_nll_loss_forward_11_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_nll_loss_forward_11_xnumel), stream=stream0)
        buf93 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.bool)
        buf94 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_12_xnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_12.run(buf1, buf93, buf94, s0, triton_poi_fused_nll_loss_backward_nll_loss_forward_12_xnumel, grid=grid(triton_poi_fused_nll_loss_backward_nll_loss_forward_12_xnumel), stream=stream0)
        buf6 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_1, _shift_logits_2], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf5, buf6, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf14 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_5, _shift_logits_6], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf13, buf14, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf22 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_9, _shift_logits_10], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf21, buf22, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf30 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_13, _shift_logits_14], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf29, buf30, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf38 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_17, _shift_logits_18], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf37, buf38, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf46 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_21, _shift_logits_22], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf45, buf46, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf54 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_25, _shift_logits_26], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf53, buf54, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf62 = empty_strided_cuda(((8 + s3) // 9, 1, 5), (5, 5*((8 + s3) // 9), 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_29, _shift_logits_30], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf61, buf62, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf70 = empty_strided_cuda((s3 + ((-8)*((8 + s3) // 9)), 1, 5), (5, ((-40)*((8 + s3) // 9)) + 5*s3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_33, _shift_logits_34], Original ATen: [aten.div, aten.tanh]
        triton_red_fused_div_tanh_13_xnumel = ((-40)*((8 + s3) // 9)) + 5*s3
        stream0 = get_raw_stream(0)
        triton_red_fused_div_tanh_13.run(buf69, buf70, triton_red_fused_div_tanh_13_xnumel, 52480, grid=grid(triton_red_fused_div_tanh_13_xnumel), stream=stream0)
        buf7 = empty_strided_cuda(((8 + s3) // 9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_1, _shift_logits_2], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf6, buf7, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_1, _shift_logits_2, cross_entropy_loss], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf5, buf7, buf8, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf15 = empty_strided_cuda(((8 + s3) // 9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_5, _shift_logits_6], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf14, buf15, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_5, _shift_logits_6, cross_entropy_loss_1], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf13, buf15, buf16, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf23 = empty_strided_cuda(((8 + s3) // 9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_9, _shift_logits_10], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf22, buf23, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_9, _shift_logits_10, cross_entropy_loss_2], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf21, buf23, buf24, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf31 = empty_strided_cuda(((8 + s3) // 9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_13, _shift_logits_14], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf30, buf31, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_13, _shift_logits_14, cross_entropy_loss_3], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf29, buf31, buf32, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf39 = empty_strided_cuda(((8 + s3) // 9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_17, _shift_logits_18], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf38, buf39, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf40 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_17, _shift_logits_18, cross_entropy_loss_4], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf37, buf39, buf40, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf47 = empty_strided_cuda(((8 + s3) // 9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_21, _shift_logits_22], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf46, buf47, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf48 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_21, _shift_logits_22, cross_entropy_loss_5], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf45, buf47, buf48, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf55 = empty_strided_cuda(((8 + s3) // 9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_25, _shift_logits_26], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf54, buf55, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf56 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_25, _shift_logits_26, cross_entropy_loss_6], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf53, buf55, buf56, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf63 = empty_strided_cuda(((8 + s3) // 9, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_29, _shift_logits_30], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf62, buf63, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf64 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_29, _shift_logits_30, cross_entropy_loss_7], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = 5*((8 + s3) // 9)
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf61, buf63, buf64, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf71 = empty_strided_cuda((s3 + ((-8)*((8 + s3) // 9)), 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_shift_logits_33, _shift_logits_34], Original ATen: [aten.div, aten.tanh]
        triton_per_fused_div_tanh_14_xnumel = s3 + ((-8)*((8 + s3) // 9))
        stream0 = get_raw_stream(0)
        triton_per_fused_div_tanh_14.run(buf70, buf71, triton_per_fused_div_tanh_14_xnumel, 5, grid=grid(triton_per_fused_div_tanh_14_xnumel), stream=stream0)
        buf72 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_33, _shift_logits_34, cross_entropy_loss_8], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_red_fused__log_softmax_div_tanh_15_xnumel = ((-40)*((8 + s3) // 9)) + 5*s3
        stream0 = get_raw_stream(0)
        triton_red_fused__log_softmax_div_tanh_15.run(buf69, buf71, buf72, triton_red_fused__log_softmax_div_tanh_15_xnumel, 52480, grid=grid(triton_red_fused__log_softmax_div_tanh_15_xnumel), stream=stream0)
        buf9 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        buf10 = reinterpret_tensor(buf9, ((8 + s3) // 9, 1), (1, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_1, _shift_logits_2, cross_entropy_loss], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf10, buf8, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf8
        buf11 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy_loss], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_17_rnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_forward_17.run(buf1, buf5, buf7, buf10, buf11, s0, 1, triton_red_fused_nll_loss_forward_17_rnumel, grid=grid(1), stream=stream0)
        buf17 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        buf18 = reinterpret_tensor(buf17, ((8 + s3) // 9, 1), (1, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_5, _shift_logits_6, cross_entropy_loss_1], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf18, buf16, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf16
        buf19 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_1], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_18_rnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_forward_18.run(buf1, buf13, buf15, buf18, buf19, s0, 1, triton_red_fused_nll_loss_forward_18_rnumel, grid=grid(1), stream=stream0)
        buf25 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        buf26 = reinterpret_tensor(buf25, ((8 + s3) // 9, 1), (1, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_9, _shift_logits_10, cross_entropy_loss_2], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf26, buf24, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf24
        buf27 = empty_strided_cuda((), (), torch.float32)
        buf89 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.bool)
        buf90 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_2], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_19_rnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_19.run(buf1, buf21, buf23, buf26, buf27, buf89, buf90, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_19_rnumel, grid=grid(1), stream=stream0)
        buf33 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        buf34 = reinterpret_tensor(buf33, ((8 + s3) // 9, 1), (1, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_13, _shift_logits_14, cross_entropy_loss_3], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf34, buf32, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf32
        buf35 = empty_strided_cuda((), (), torch.float32)
        buf87 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.bool)
        buf88 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_3], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_20_rnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_20.run(buf1, buf29, buf31, buf34, buf35, buf87, buf88, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_20_rnumel, grid=grid(1), stream=stream0)
        buf41 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        buf42 = reinterpret_tensor(buf41, ((8 + s3) // 9, 1), (1, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_17, _shift_logits_18, cross_entropy_loss_4], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf42, buf40, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf40
        buf43 = empty_strided_cuda((), (), torch.float32)
        buf85 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.bool)
        buf86 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_4], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_21_rnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_21.run(buf1, buf37, buf39, buf42, buf43, buf85, buf86, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_21_rnumel, grid=grid(1), stream=stream0)
        buf49 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        buf50 = reinterpret_tensor(buf49, ((8 + s3) // 9, 1), (1, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_21, _shift_logits_22, cross_entropy_loss_5], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf50, buf48, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf48
        buf51 = empty_strided_cuda((), (), torch.float32)
        buf83 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.bool)
        buf84 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_5], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_22_rnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_22.run(buf1, buf45, buf47, buf50, buf51, buf83, buf84, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_22_rnumel, grid=grid(1), stream=stream0)
        buf57 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        buf58 = reinterpret_tensor(buf57, ((8 + s3) // 9, 1), (1, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_25, _shift_logits_26, cross_entropy_loss_6], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf58, buf56, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf56
        buf59 = empty_strided_cuda((), (), torch.float32)
        buf81 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.bool)
        buf82 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_6], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_23_rnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_23.run(buf1, buf53, buf55, buf58, buf59, buf81, buf82, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_23_rnumel, grid=grid(1), stream=stream0)
        buf65 = empty_strided_cuda(((8 + s3) // 9, 1), (1, (8 + s3) // 9), torch.float32)
        buf66 = reinterpret_tensor(buf65, ((8 + s3) // 9, 1), (1, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_29, _shift_logits_30, cross_entropy_loss_7], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = (8 + s3) // 9
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf66, buf64, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf64
        buf67 = empty_strided_cuda((), (), torch.float32)
        buf79 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.bool)
        buf80 = empty_strided_cuda(((8 + s0) // 9, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [cross_entropy_loss, cross_entropy_loss_7], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        triton_red_fused_nll_loss_backward_nll_loss_forward_24_rnumel = (8 + s0) // 9
        stream0 = get_raw_stream(0)
        triton_red_fused_nll_loss_backward_nll_loss_forward_24.run(buf1, buf61, buf63, buf66, buf67, buf79, buf80, s0, 1, triton_red_fused_nll_loss_backward_nll_loss_forward_24_rnumel, grid=grid(1), stream=stream0)
        buf73 = empty_strided_cuda((s3 + ((-8)*((8 + s3) // 9)), 1), (1, s3 + ((-8)*((8 + s3) // 9))), torch.float32)
        buf74 = reinterpret_tensor(buf73, (s3 + ((-8)*((8 + s3) // 9)), 1), (1, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [_shift_logits_33, _shift_logits_34, cross_entropy_loss_8], Original ATen: [aten.div, aten.tanh, aten._log_softmax]
        triton_per_fused__log_softmax_div_tanh_16_xnumel = s3 + ((-8)*((8 + s3) // 9))
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_div_tanh_16.run(buf74, buf72, triton_per_fused__log_softmax_div_tanh_16_xnumel, 5, grid=grid(triton_per_fused__log_softmax_div_tanh_16_xnumel), stream=stream0)
        del buf72
        buf77 = empty_strided_cuda((s0 + ((-8)*((8 + s0) // 9)), 1), (1, 1), torch.bool)
        buf78 = empty_strided_cuda((s0 + ((-8)*((8 + s0) // 9)), 1), (1, 1), torch.int64)
        buf76 = empty_strided_cuda((), (), torch.float32)
        buf95 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_loss, loss, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, cross_entropy_loss_8, loss_8, tensor, loss_9], Original ATen: [aten.nll_loss_forward, aten.add, aten._to_copy, aten.div, aten.nll_loss_backward]
        triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_25_rnumel = s0 + ((-8)*((8 + s0) // 9))
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_25.run(buf95, buf1, buf69, buf71, buf74, primals_8, buf19, buf27, buf35, buf43, buf51, buf59, buf67, buf77, buf78, buf76, s0, 1, triton_red_fused__to_copy_add_div_nll_loss_backward_nll_loss_forward_25_rnumel, grid=grid(1), stream=stream0)
        del buf1
        del buf19
        del buf27
        del buf35
        del buf43
        del buf51
        del buf59
        del buf67
        del primals_8
    return (buf95, buf5, buf7, buf10, buf13, buf15, buf18, buf21, buf23, buf26, buf29, buf31, buf34, buf37, buf39, buf42, buf45, buf47, buf50, buf53, buf55, buf58, buf61, buf63, buf66, buf69, buf71, buf74, buf76, buf77, buf78, primals_1, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, s3, s0, s3 + ((-8)*((8 + s3) // 9)), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((262400, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = 533
    primals_3 = rand_strided((1, 533), (533, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = 533
    primals_5 = rand_strided((1, 533), (533, 1), device='cuda:0', dtype=torch.int64)
    primals_6 = 533
    primals_7 = rand_strided((1, 533, 2048), (1091584, 2048, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
