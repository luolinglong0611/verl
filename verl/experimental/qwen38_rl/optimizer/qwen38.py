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
"""Install the BF16 streaming Adam path for the loaded TP8/EP64 actor."""

from datetime import timedelta

from ..megatron_loader import TRAIN_LOCAL_EXPERTS, _world_gate, capture_groups


def optimizer_coordinates(binding, groups):
    from .coordinates import SRCoordinates

    parameter = binding.parameter
    if binding.expert is not None:
        coordinate = binding.expert
        begin = groups.ep_rank * TRAIN_LOCAL_EXPERTS
        if not begin <= coordinate.global_expert < begin + TRAIN_LOCAL_EXPERTS:
            raise ValueError(f"expert update identity differs from EP ownership at {binding.name}")
        key = f"{coordinate.hf_name}/global-expert-{coordinate.global_expert}"
    elif getattr(parameter, "tensor_model_parallel", False):
        key = f"{binding.name}/tp8-shard-{groups.tp_rank}"
    else:
        key = f"{binding.name}/tp8-replica"
    shape = tuple(parameter.shape)
    return SRCoordinates(key, shape, shape).validate()


def _bind(chunks):
    import torch
    from megatron.bridge.models.conversion.utils import unwrap_model
    from megatron.core.distributed import DistributedDataParallel

    groups = capture_groups()
    model = unwrap_model(chunks)
    if len(chunks) != 1 or len(model) != 1 or not isinstance(chunks[0], DistributedDataParallel):
        raise ValueError("expected one live Megatron DDP actor")
    config = chunks[0].ddp_config
    if config.grad_reduce_in_fp32 or config.use_distributed_optimizer or config.use_megatron_fsdp:
        raise ValueError("expected BF16 DDP main gradients and nonsharded optimizer ownership")
    dense, experts = model[0]._q38_import_bindings
    expected = {id(p) for p in chunks[0].parameters() if p.requires_grad}
    actual = {id(binding.parameter) for binding in dense + experts}
    if actual != expected or len(actual) != len(dense) + len(experts):
        raise ValueError("optimizer ownership differs from the verified checkpoint destinations")
    for binding in dense + experts:
        parameter = binding.parameter
        gradient = getattr(parameter, "main_grad", None)
        if (
            parameter.dtype != torch.bfloat16
            or not parameter.is_cuda
            or gradient is None
            or gradient.dtype != torch.bfloat16
            or gradient.device != parameter.device
            or gradient.shape != parameter.shape
        ):
            raise ValueError(f"live BF16 parameter/main gradient mismatch at {binding.name}")
        parameter._q38_sr_coordinates = optimizer_coordinates(binding, groups)


class Qwen38Bf16OptimizerFactory:
    """Construct the TP8/EP64 actor optimizer with a Gloo commit vote."""

    @staticmethod
    def build(*, model, config):
        """Bind checkpoint coordinates before constructing the live optimizer."""
        import torch
        import torch.distributed as dist
        from megatron.core import optimizer as core_optimizer

        from . import megatron_factory as factory
        from .streaming_coordinates import StreamingBf16AdamWCoordinates
        from .update_witness import attach

        if (
            config.use_distributed_optimizer
            or config.fp16
            or not config.bf16
            or config.use_precision_aware_optimizer
            or not config.optimizer_cpu_offload
        ):
            raise ValueError("expected nonsharded BF16 Adam with CPU moment storage")

        config.main_params_dtype = config.main_grads_dtype = torch.bfloat16
        config.exp_avg_dtype = config.exp_avg_sq_dtype = torch.bfloat16
        config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True
        config.q38_bf16_hdo = config.q38_nonsharded_bf16 = True
        factory._check_config(config)

        error = None
        try:
            _bind(model)
        except Exception as exc:
            error = exc
        _world_gate(error, "BF16 optimizer ownership")
        if dist.get_world_size() != 64 or not dist.is_gloo_available():
            raise RuntimeError("Qwen3.8 Adam requires the 64-rank CPU commit group")
        config.q38_cpu_commit_group = dist.new_group(
            ranks=list(range(64)), backend="gloo", timeout=timedelta(minutes=45)
        )

        if getattr(core_optimizer, "_q38_candidate_installed", False):
            if getattr(core_optimizer.get_megatron_optimizer, "_q38_factory_module", None) != factory.__name__:
                raise RuntimeError("a different BF16 Megatron optimizer is already installed")
        factory.StreamingBf16AdamW = StreamingBf16AdamWCoordinates
        factory.install()
        optimizer = core_optimizer.get_megatron_optimizer(config=config, model_chunks=model)
        return attach(optimizer, model, torch, dist)
