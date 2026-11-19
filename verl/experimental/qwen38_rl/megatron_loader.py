# Copyright 2026 the contributors to this fork
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rank-local Qwen3.8 BF16 import through the pinned Megatron Bridge."""

import hashlib
import os
import re
import socket
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from verl.utils.device import get_device_id, get_torch_device

from .remote_checkpoint import (
    DOWN_SHAPE,
    GATE_SHAPE,
    LAYERS,
    TRAIN_EP,
    TRAIN_TP,
    Checkpoint,
    DenseBridgeState,
    Metadata,
    decode_json,
)

EXPERT_PARAMETER = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)\Z")
TRAIN_LOCAL_EXPERTS = 8
TRAIN_VOCAB_DIVISOR = 64


@dataclass(frozen=True)
class ExpertCoordinate:
    layer: int
    field: str
    local_slot: int
    global_expert: int

    @property
    def hf_name(self):
        return f"model.layers.{self.layer}.mlp.experts.{self.field}"

    @property
    def shape(self):
        return GATE_SHAPE if self.field == "gate_up_proj" else DOWN_SHAPE


def expert_coordinate(local_name, bridge_name, ep_rank):
    local = EXPERT_PARAMETER.fullmatch(local_name)
    if local is None:
        raise ValueError(f"unexpected grouped expert parameter {local_name}")
    layer, fc, slot = map(int, local.groups())
    if not 0 <= layer < LAYERS or not 0 <= ep_rank < TRAIN_EP or not 0 <= slot < TRAIN_LOCAL_EXPERTS:
        raise ValueError(f"out-of-range expert parameter {local_name}")
    expert = ep_rank * TRAIN_LOCAL_EXPERTS + slot
    expected_bridge_name = f"decoder.layers.{layer}.mlp.experts.linear_fc{fc}.weight{expert}"
    if bridge_name != expected_bridge_name:
        raise ValueError(f"Bridge source identity disagrees with live EP ownership at {local_name}")
    return ExpertCoordinate(layer, "gate_up_proj" if fc == 1 else "down_proj", slot, expert)


@dataclass(frozen=True)
class Groups:
    rank: int
    tp_rank: int
    ep_rank: int
    dp_rank: int
    tp_group: object
    dp_group: object


def capture_groups():
    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    rank = dist.get_rank()
    tp_rank = rank % TRAIN_TP
    dp_rank = rank // TRAIN_TP
    tp = ps.get_tensor_model_parallel_group()
    dp = ps.get_data_parallel_group(with_context_parallel=True)
    pp = ps.get_pipeline_model_parallel_group()
    ep = ps.get_expert_model_parallel_group()
    etp = ps.get_expert_tensor_parallel_group()
    edp = ps.get_expert_data_parallel_group()
    actual = tuple(dist.get_process_group_ranks(group) for group in (tp, dp, pp, ep, etp, edp))
    expected = (
        list(range(dp_rank * TRAIN_TP, (dp_rank + 1) * TRAIN_TP)),
        list(range(tp_rank, TRAIN_EP, TRAIN_TP)),
        [rank],
        list(range(TRAIN_EP)),
        [rank],
        [rank],
    )
    if dist.get_world_size() != TRAIN_EP or actual != expected or ps.get_expert_model_parallel_rank() != rank:
        raise ValueError("actual TP8/DP8/PP1/EP64/ETP1/EDP1 ranks differ from the checkpoint plan")
    return Groups(rank, tp_rank, rank, dp_rank, tp, dp)


@dataclass
class Binding:
    name: str
    parameter: object
    task: object
    sources: tuple
    expert: ExpertCoordinate | None


def task_sources(task):
    source = task.mapping.hf_param
    return (source,) if isinstance(source, str) else tuple(source.values())


def build_bindings(bridge, hf_config, model, state, groups, *, layer_count=LAYERS):
    import torch

    expected_experts = list(range(groups.ep_rank * TRAIN_LOCAL_EXPERTS, (groups.ep_rank + 1) * TRAIN_LOCAL_EXPERTS))
    if len(model.decoder.layers) != layer_count or any(
        layer.mlp.local_expert_indices != expected_experts for layer in model.decoder.layers
    ):
        raise ValueError("Megatron's live layer expert indices differ from the checkpoint plan")
    hf = SimpleNamespace(config=hf_config, state=state, model_name_or_path="Qwen/Qwen3.8-2.4T-A95B")
    tasks = bridge.build_conversion_tasks(hf, [model])
    live = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    seen = set()
    dense = []
    experts = []
    for task in tasks:
        if task is None or task.param_weight is None or task.megatron_module is None:
            raise ValueError("incomplete Bridge conversion task for PP1")
        parameter = task.param_weight
        identity = id(parameter)
        if identity not in live or identity in seen or task.param_name != live[identity]:
            raise ValueError(f"Bridge task does not bind exactly one live parameter {task.param_name}")
        if parameter.dtype != torch.bfloat16 or not parameter.is_cuda or not parameter.is_contiguous():
            raise ValueError(f"live parameter must have contiguous CUDA BF16 storage {task.param_name}")
        sources = task_sources(task)
        if not sources or not all(source in state for source in sources):
            raise ValueError(f"unknown pinned Bridge input at {task.param_name}")
        seen.add(identity)
        if not getattr(parameter, "allreduce", True):
            coordinate = expert_coordinate(task.param_name, task.global_param_name, groups.ep_rank)
            mapping = "FusedGatedExpertMapping" if coordinate.field == "gate_up_proj" else "FusedExpertMapping"
            if (
                sources != (coordinate.hf_name,)
                or tuple(parameter.shape) != coordinate.shape
                or type(task.mapping).__name__ != mapping
                or task.mapping.megatron_param != task.global_param_name
                or task.mapping.tp_size != 1
            ):
                raise ValueError(f"live ETP1 expert layout disagrees with source at {task.param_name}")
            experts.append(Binding(task.param_name, parameter, task, sources, coordinate))
        else:
            if any(".mlp.experts." in name for name in sources):
                raise ValueError(f"unexpected expert source bound to dense parameter {task.param_name}")
            if task.mapping.tp_size != TRAIN_TP or task.mapping.tp_rank != groups.tp_rank:
                raise ValueError(f"dense Bridge mapping does not own the correct TP8 rank {task.param_name}")
            dense.append(Binding(task.param_name, parameter, task, sources, None))
    if seen != set(live):
        raise ValueError("Bridge tasks do not cover the full live policy")
    dense.sort(key=lambda binding: binding.name)
    experts.sort(key=lambda binding: (binding.expert.layer, binding.expert.field, binding.expert.local_slot))
    source_names = [name for entry in dense for name in entry.sources]
    expected_dense = {name for name in state if ".mlp.experts." not in name}
    if set(source_names) != expected_dense or len(source_names) != len(expected_dense):
        raise ValueError("dense Bridge tasks must consume every pinned dense source once")
    coordinates = {(entry.expert.layer, entry.expert.field, entry.expert.local_slot) for entry in experts}
    if len(coordinates) != len(experts) or len(experts) != layer_count * TRAIN_LOCAL_EXPERTS * 2:
        raise ValueError("incomplete ETP1 source coverage")
    return dense, experts


def _world_gate(error, label):
    import torch
    import torch.distributed as dist

    ready = torch.tensor(int(error is None), dtype=torch.int32, device=get_device_id())
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not bool(ready.item()):
        raise RuntimeError(f"checkpoint import failed on at least one training rank during {label}") from error


def _replicated_placeholder(binding, checkpoint):
    """Supply shape-only input for Bridge delegates that read on all TP ranks."""
    import torch

    mapping = binding.task.mapping
    if type(mapping).__name__ not in ("ReplicatedMapping", "RMSNorm2ZeroCenteredRMSNormMapping") and getattr(
        binding.parameter, "tensor_model_parallel", False
    ):
        return None
    if len(binding.sources) != 1:
        raise ValueError(f"unexpected replicated Bridge task {binding.name}")
    shape = checkpoint.locations[binding.sources[0]].shape
    if shape != tuple(binding.parameter.shape):
        raise ValueError(f"replicated Bridge shape mismatch at {binding.name}")
    return torch.empty(shape, dtype=torch.bfloat16, device="cpu")


def _expert_read_order(experts, checkpoint, rank):
    """Round-robin mounted source hosts with a distinct rank starting point."""
    by_host = {}
    for entry in experts:
        shard = checkpoint.locations[entry.expert.hf_name].shard
        host, _ = checkpoint.endpoints[shard]
        by_host.setdefault(host, deque()).append(entry)
    hosts = sorted(by_host)
    shift = rank % len(hosts)
    hosts = hosts[shift:] + hosts[:shift]
    while hosts:
        for host in hosts:
            yield by_host[host].popleft()
        hosts = [host for host in hosts if by_host[host]]


def broadcast_dense_parameter(binding, groups):
    import torch.distributed as dist

    dist.broadcast(binding.parameter.detach(), src=groups.tp_rank, group=groups.dp_group)


def load_policy(bridge, hf_config, model, checkpoint):
    import torch
    import torch.distributed as dist

    state = DenseBridgeState(checkpoint)
    error = None
    try:
        groups = capture_groups()
        dense, experts = build_bindings(bridge, hf_config, model, state, groups)
    except Exception as exc:
        error = exc
    _world_gate(error, "provider binding")

    schedule = [(entry.name, entry.sources, tuple(entry.parameter.shape)) for entry in dense]
    signature = hashlib.sha256(repr(schedule).encode()).hexdigest()
    candidates = [None] * dist.get_world_size(groups.dp_group)
    dist.all_gather_object(candidates, signature, group=groups.dp_group)
    _world_gate(
        None if len(set(candidates)) == 1 else ValueError("dense DP schedules differ"), "dense order validation"
    )

    error = None
    expert_receipts = []
    try:
        for entry in _expert_read_order(experts, checkpoint, groups.rank):
            coordinate = entry.expert
            digest = checkpoint.read_into(
                coordinate.hf_name, coordinate.global_expert, coordinate.global_expert + 1, entry.parameter.unsqueeze(0)
            )
            expert_receipts.append((coordinate.hf_name, coordinate.global_expert, digest))
        get_torch_device().synchronize()
    except Exception as exc:
        error = exc
    _world_gate(error, "expert ranges")

    for entry in dense:
        source = None
        error = None
        if groups.dp_rank == 0:
            try:
                if groups.rank == 0:
                    source = bridge.maybe_modify_loaded_hf_weight(entry.task.mapping.hf_param, state)
                else:
                    source = _replicated_placeholder(entry, checkpoint)
            except Exception as exc:
                error = exc
        _world_gate(error, entry.name + " source")
        if groups.dp_rank == 0:
            try:
                task = entry.task
                converted = task.mapping.hf_to_megatron(source, task.megatron_module)
                if converted is None or converted.shape != entry.parameter.shape or converted.dtype != torch.bfloat16:
                    raise ValueError(f"unexpected converted shape or dtype at {entry.name}")
                with torch.no_grad():
                    entry.parameter.copy_(converted)
                del converted
            except Exception as exc:
                error = exc
        del source
        _world_gate(error, entry.name + " conversion")
        broadcast_dense_parameter(entry, groups)

    expected_reads = {name for entry in dense for name in entry.sources} if groups.rank == 0 else set()
    _world_gate(None if state.seen == expected_reads else ValueError("dense reads differ from plan"), "source coverage")
    model._q38_import_bindings = (dense, experts)
    return {
        "rank": groups.rank,
        "expert_count": len(expert_receipts),
        "expert_source_sha256": hashlib.sha256(repr(expert_receipts).encode()).hexdigest(),
        "dense_parameter_count": len(dense),
        "dense_sources_read": len(state.seen),
        "dense_source_peak_bytes": state.max_live_bytes,
        "local_trainable_bf16_bytes": sum(2 * item.parameter.numel() for item in dense + experts),
        "checkpoint_bytes_read": checkpoint.bytes_read,
    }


def iter_local_expert_weights(model):
    """Emit live canonical HF weights for this rank's eight experts."""
    import torch

    _, experts = model._q38_import_bindings
    live = dict(model.named_parameters())
    expert_groups = {}
    for entry in experts:
        if live.get(entry.name) is not entry.parameter:
            raise ValueError(f"live parameter identity changed at {entry.name}")
        coordinate = entry.expert
        expert_groups.setdefault(coordinate.hf_name, []).append((coordinate, entry.parameter))
    for name, items in expert_groups.items():
        slots = [coordinate.local_slot for coordinate, _ in items]
        if slots != list(range(TRAIN_LOCAL_EXPERTS)):
            raise ValueError(f"invalid ETP1 export order at {name}")
        yield name, torch.stack([parameter.detach() for _, parameter in items])


def iter_dense_weights(model):
    """All eight ranks of DP0 participate, while rank zero yields dense HF weights."""
    import torch

    groups = capture_groups()
    if groups.dp_rank != 0:
        return
    dense, _ = model._q38_import_bindings
    live = dict(model.named_parameters())
    for entry in dense:
        if live.get(entry.name) is not entry.parameter:
            raise ValueError(f"live parameter identity changed at {entry.name}")
        task = entry.task
        with torch.no_grad():
            tensors = task.mapping.megatron_to_hf(entry.parameter.detach(), task.megatron_module)
        if groups.rank == 0:
            if set(tensors) != set(entry.sources):
                raise ValueError(f"dense Bridge export source mismatch at {entry.name}")
            for name in entry.sources:
                weight = tensors[name]
                if name.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")):
                    weight = weight.float()
                yield name, weight.contiguous()


def _checkpoint_from_env(path):
    metadata = Metadata.from_files(path, os.environ["Q38_OBJECT_MANIFEST"], os.environ["Q38_POLICY_MANIFEST"])
    endpoints = decode_json(Path(os.environ["Q38_CHECKPOINT_ENDPOINTS"]).read_bytes())
    local_ip = socket.gethostbyname(socket.gethostname())
    return Checkpoint.from_catalog(
        metadata,
        os.environ["Q38_CHECKPOINT_CATALOG"],
        os.environ["Q38_CHECKPOINT_CATALOG_SHA256"],
        endpoints,
        local_ip=local_ip,
    )
