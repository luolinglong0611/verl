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
"""Opt-in nonshared research adapter for MCore 0.19.2; not production-qualified.

Install before calling get_megatron_optimizer and set config.q38_bf16_hdo=True.
All ranks must call the same chained optimizer step sequence. Fused kernels that
bypass normal autograd parameter hooks are unsupported: parameter activity
must be qualified for the actual model and kernel stack before deployment.
"""

import functools
import hashlib
import json
import math
import time
import types

import torch
import torch.distributed as dist

from verl.utils.device import get_device_id, get_torch_device

from .finite import batched_finite
from .route_activity import RoutedActivity
from .streaming import StreamingBf16AdamW


class _UnallocatedPlaceholder(torch.optim.AdamW):
    """Factory's pre-sharding optimizer; makes no master or optimizer state."""

    def __init__(self, params, **kw):
        super().__init__(params, lr=kw["lr"], weight_decay=kw["weight_decay"], betas=kw["betas"], eps=kw["eps"])


def _check_config(config):
    nonshard = getattr(config, "q38_nonsharded_bf16", False)
    if not (
        config.optimizer == "adam"
        and config.bf16
        and config.optimizer_cpu_offload
        and config.optimizer_offload_fraction == 1.0
        and config.main_params_dtype == config.main_grads_dtype == torch.bfloat16
        and config.exp_avg_dtype == config.exp_avg_sq_dtype == torch.bfloat16
    ):
        raise ValueError("requires BF16 model/gradient/moments, Adam and full CPU offload")
    if nonshard:
        if config.use_distributed_optimizer or not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            raise ValueError("requires nonsharded BF16 decoupled gradients")
    elif not (config.use_distributed_optimizer and config.use_precision_aware_optimizer):
        raise ValueError("requires distributed precision-aware optimizer")
    if config.overlap_cpu_optimizer_d2h_h2d or config.overlap_param_gather_with_optimizer_step:
        raise ValueError("overlap not qualified")


def cpu_step_commit(success, group):
    """Require the local Adam result from every rank before publication."""
    ok = torch.tensor(success, dtype=torch.int32, device="cpu")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    return bool(ok.item())


def install():
    import megatron.core.optimizer as factory
    from megatron.core import parallel_state as ps
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.tensor_parallel.layers import param_is_not_tensor_parallel_duplicate
    from megatron.core.transformer.module import param_is_not_shared
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    from .nonsharded import NonShardedBf16Optimizer

    if getattr(factory, "_q38_candidate_installed", False):
        return
    original_factory = factory._get_megatron_optimizer_based_on_param_groups
    original_get = factory.get_megatron_optimizer
    globals_copy = original_factory.__globals__.copy()
    globals_copy["HybridDeviceOptimizer"] = _UnallocatedPlaceholder
    candidate_factory = types.FunctionType(
        original_factory.__code__,
        globals_copy,
        original_factory.__name__,
        original_factory.__defaults__,
        original_factory.__closure__,
    )
    candidate_factory.__kwdefaults__ = original_factory.__kwdefaults__
    globals_nonshard = globals_copy.copy()
    globals_nonshard["Float16OptimizerWithFloat16Params"] = NonShardedBf16Optimizer
    nonshard_factory = types.FunctionType(
        original_factory.__code__,
        globals_nonshard,
        original_factory.__name__,
        original_factory.__defaults__,
        original_factory.__closure__,
    )
    nonshard_factory.__kwdefaults__ = original_factory.__kwdefaults__

    @functools.wraps(original_factory)
    def opt_in_factory(config, model_chunks, *args, **kwargs):
        if not getattr(config, "q38_bf16_hdo", False):
            return original_factory(config, model_chunks, *args, **kwargs)
        _check_config(config)
        nonshard = getattr(config, "q38_nonsharded_bf16", False)
        for chunk in model_chunks:
            if (
                chunk.ddp_config.grad_reduce_in_fp32
                or chunk.ddp_config.use_distributed_optimizer == nonshard
                or getattr(chunk.ddp_config, "use_megatron_fsdp", False)
            ):
                raise ValueError("requires Megatron BF16 DDP gradient buffers")
        result = (nonshard_factory if nonshard else candidate_factory)(config, model_chunks, *args, **kwargs)
        if not isinstance(result, NonShardedBf16Optimizer if nonshard else DistributedOptimizer):
            raise ValueError("only distributed optimizer is supported")
        if result.is_stub_optimizer:
            return result
        names = {}
        for i, chunk in enumerate(model_chunks):
            for name, param in chunk.named_parameters():
                names[id(param)] = f"chunk{i}/{name}"
        if nonshard:
            pairs = [([p], [p]) for group in result.optimizer.param_groups for p in group["params"]]
        else:
            pairs = zip(result.model_float16_groups, result.shard_float16_groups, strict=False)
        for model_group, shard_group in pairs:
            for model_param, shard in zip(model_group, shard_group, strict=False):
                name = names.get(id(model_param))
                if name is None:
                    raise ValueError("model shard has no stable parameter name")
                expert = not getattr(model_param, "allreduce", True)
                # Key stability requires a fixed parallel layout. DP replicas
                # deliberately receive the same key and nonoverlap offsets.
                shard._q38_canonical_name = (
                    f"{name}/pp{ps.get_pipeline_model_parallel_rank()}"
                    f"/tp{ps.get_tensor_model_parallel_rank()}"
                    + (f"/ep{ps.get_expert_model_parallel_rank()}" if expert else "")
                )
                shard._q38_global_offset = (
                    0 if nonshard else result._get_model_param_range_map(model_param)["param"].start
                )
                if shard.dtype != torch.bfloat16:
                    raise ValueError("trainable parameters must be BF16")
        if not nonshard and any(result.model_fp32_groups):
            raise ValueError("FP32 trainable parameters are not qualified")
        result.optimizer = StreamingBf16AdamW(
            result.optimizer.param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            require_finite_ticket=True,
        )
        return result

    @functools.wraps(original_get)
    def opt_in_get(config, model_chunks, *args, **kwargs):
        result = original_get(config, model_chunks, *args, **kwargs)
        if not getattr(config, "q38_bf16_hdo", False):
            return result
        _check_config(config)
        if not hasattr(result, "chained_optimizers"):
            raise ValueError("candidate expects a chained optimizer")
        records, seen, observers, routed_ids = [], {}, [], set()
        for chunk in model_chunks:
            for module in chunk.modules():
                if isinstance(module, TEGroupedMLP):
                    observer = RoutedActivity(module, [module.linear_fc1, module.linear_fc2])
                    if routed_ids.intersection(observer.params):
                        raise ValueError("expert parameter belongs to multiple routed modules")
                    routed_ids.update(observer.params)
                    observers.append(observer)
        for i, chunk in enumerate(model_chunks):
            for name, param in chunk.named_parameters():
                if not param.requires_grad:
                    continue
                marker = id(param)
                if marker in seen:
                    continue
                seen[marker] = False
                records.append((f"chunk{i}/{name}", param, not getattr(param, "allreduce", True)))
                if marker not in routed_ids:
                    if not getattr(param, "allreduce", True) and not getattr(
                        config, "q38_test_allow_expert_autograd_hooks", False
                    ):
                        raise ValueError("expert parameter lacks a qualified routing observer")

                    def mark(grad, marker=marker):
                        seen[marker] = True
                        return grad

                    param.register_hook(mark)
        grouped_records = {}
        for expert in (False, True):
            group_records = [(n, p) for n, p, e in records if e == expert]
            group = (
                ps.get_expert_data_parallel_group()
                if expert
                else ps.get_data_parallel_group(with_context_parallel=True)
            )
            schema = [(n, list(p.shape), str(p.dtype)) for n, p in group_records]
            signature = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
            gathered = [None] * dist.get_world_size(group)
            dist.all_gather_object(gathered, (len(schema), signature), group=group)
            if any(other != (len(schema), signature) for other in gathered):
                raise RuntimeError("DP activity parameter schema differs across ranks")
            grouped_records[expert] = (group, group_records)
        original_zero = result.zero_grad
        original_prepare = result.prepare_grads
        nonshard = getattr(config, "q38_nonsharded_bf16", False)

        def zero(self, set_to_none=True):
            for marker in seen:
                seen[marker] = False
            for observer in observers:
                observer.reset()
            for part in self.chained_optimizers:
                if not getattr(part, "is_stub_optimizer", False):
                    part.optimizer._finite_ticket = False
            return original_zero(set_to_none)

        def prepare(self):
            found = bool(original_prepare())
            for observer in observers:
                seen.update(observer.flags())
            activity = {}
            for expert in (False, True):
                group, group_records = grouped_records[expert]
                if group_records:
                    flags = torch.tensor(
                        [seen[id(p)] for _, p in group_records], dtype=torch.int32, device=get_device_id()
                    )
                    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
                    for (_, p), active in zip(group_records, flags.cpu().tolist(), strict=False):
                        activity[id(p)] = bool(active)
            gradients = []
            for part in self.chained_optimizers:
                if getattr(part, "is_stub_optimizer", False):
                    continue
                part.optimizer._finite_ticket = False
                pairs = (
                    (([p], [p]) for p in part.get_parameters())
                    if nonshard
                    else zip(part.model_float16_groups, part.shard_float16_groups, strict=False)
                )
                for models, shards in pairs:
                    for model, shard in zip(models, shards, strict=False):
                        if not activity[id(model)]:
                            shard.decoupled_grad = None
                        grad = shard.decoupled_grad
                        if grad is None:
                            continue
                        if grad.dtype != torch.bfloat16:
                            raise ValueError("actual gradient shard is not BF16")
                        gradients.append(grad)
            finite = batched_finite(gradients)
            if found:
                finite.zero_()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            passed = bool(finite.item())
            if passed:
                for part in self.chained_optimizers:
                    if not getattr(part, "is_stub_optimizer", False):
                        part.optimizer.arm_finite_ticket()
            return not passed

        result.zero_grad = types.MethodType(zero, result)
        result.prepare_grads = types.MethodType(prepare, result)
        if nonshard:
            from megatron.core.optimizer import clip_grads as clip_module
            from megatron.core.optimizer import optimizer as optimizer_module

            from .bounded_clip import install as install_bounded_clip
            from .bounded_clip import scope as bounded_clip_scope

            install_bounded_clip(optimizer_module, clip_module)

            def grad_norm(self):
                # All physical training ranks participate. Dense TP shards
                # repeat over DP; experts repeat over expert-DP. Model TP
                # metadata filters tensors replicated within each TP group.
                sums = torch.zeros(2, dtype=torch.float64, device=get_device_id())
                buckets = [[], []]
                for part in self.chained_optimizers:
                    if getattr(part, "is_stub_optimizer", False):
                        continue
                    for p in part.get_parameters():
                        grad = getattr(p, "decoupled_grad", None)
                        if grad is None or not param_is_not_shared(p):
                            continue
                        if not param_is_not_tensor_parallel_duplicate(
                            p,
                            tp_group=ps.get_tensor_model_parallel_group(),
                            expert_tp_group=ps.get_expert_tensor_parallel_group(),
                        ):
                            continue
                        buckets[int(id(p) in routed_ids)].append(grad)

                def flush_norm(batch, category):
                    if batch:
                        norms = torch._foreach_norm(batch, 2.0, dtype=torch.float32)
                        sums[category] += torch.stack(norms).to(torch.float64).square().sum()
                        batch.clear()

                for category, gradients in enumerate(buckets):
                    batch = []
                    for grad in gradients:
                        flat = grad.reshape(-1)
                        for start in range(0, flat.numel(), 1048576):
                            batch.append(flat[start : start + 1048576])
                            if len(batch) == 32:
                                flush_norm(batch, category)
                    flush_norm(batch, category)
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                value = math.sqrt(
                    float(sums[0]) / ps.get_data_parallel_world_size(with_context_parallel=True)
                    + float(sums[1]) / ps.get_expert_data_parallel_world_size()
                )
                if not math.isfinite(value):
                    raise FloatingPointError("nonfinite global gradient norm")
                return value

            original_step = result.step
            cpu_commit_group = getattr(config, "q38_cpu_commit_group", None)

            def guarded_step(self):
                # Independent phase-separated inference has no shared reader
                # lease. Preserve a world-level outcome vote after the actual
                # optimizer and its CUDA writes complete.
                error, outcome = None, None
                started = time.monotonic()
                try:
                    with bounded_clip_scope():
                        outcome = original_step()
                    get_torch_device().synchronize()
                except Exception as exc:
                    error = exc
                if cpu_commit_group is not None:
                    # The pageable CPU Adam tiles can finish minutes apart.
                    # Keep CUDA synchronized locally and rendezvous on the CPU
                    # so a faster rank does not hold an in-flight GPU collective.
                    success = error is None and bool(outcome and outcome[0])
                    print(
                        "q38_optimizer_local_update "
                        + json.dumps(
                            {
                                "rank": dist.get_rank(),
                                "success": success,
                                "error_type": None if error is None else type(error).__name__,
                                "seconds": time.monotonic() - started,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    committed = cpu_step_commit(success, cpu_commit_group)
                else:
                    ok = torch.tensor(error is None, dtype=torch.int32, device=get_device_id())
                    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                    committed = bool(ok.item())
                if not committed:
                    raise RuntimeError("optimizer failed on at least one rank") from error
                return outcome

            result.get_grad_norm = types.MethodType(grad_norm, result)
            result.step = types.MethodType(guarded_step, result)
        return result

    factory._get_megatron_optimizer_based_on_param_groups = opt_in_factory
    opt_in_get._q38_factory_module = __name__
    factory.get_megatron_optimizer = opt_in_get
    factory._q38_candidate_installed = True
