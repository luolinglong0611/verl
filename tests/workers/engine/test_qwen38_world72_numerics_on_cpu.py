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
"""CPU numerical qualification with an independent NumPy/Python reference."""

import math

import numpy as np
import torch

from examples.rl_rollout.world72_training.qwen38_bf16_optimizer import (
    Bf16CpuAdamW,
    CpuBf16Moments,
    GradientStore,
    counter_random16,
    rounding_key,
    stochastic_bf16,
)

MASK = (1 << 64) - 1


def reference_uniform16(key, offset, n):
    result = []
    for coordinate in range(offset, offset + n):
        value = (coordinate + key + 0x9E3779B97F4A7C15) & MASK
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        value ^= value >> 31
        result.append(value >> 48)
    return np.asarray(result, dtype=np.uint64)


def reference_round(value, key, offset=0):
    values = np.asarray(value, dtype=np.float32)
    bits = values.view(np.uint32).reshape(-1)
    up = ((bits >> 16).astype(np.uint64) + (reference_uniform16(key, offset, bits.size) < (bits & 65535))).astype(
        np.uint16
    )
    return (up.astype(np.uint32) << 16).view(np.float32).reshape(values.shape)


def numpy_adam(weight, mean, variance, grad, *, step, lr, betas, eps, wd, name, seed, return_unrounded=False):
    b1, b2 = betas
    weight = np.multiply(weight, np.float32(1 - lr * wd), dtype=np.float32)
    diff = np.subtract(grad, mean, dtype=np.float32)
    mean = np.asarray(mean.astype(np.float64) + float(np.float32(1 - b1)) * diff.astype(np.float64), dtype=np.float32)
    variance = np.multiply(variance, np.float32(b2), dtype=np.float32)
    variance = np.asarray(
        variance.astype(np.float64) + float(np.float32(1 - b2)) * grad.astype(np.float64) ** 2, dtype=np.float32
    )
    denom = np.sqrt(variance, dtype=np.float32)
    denom = np.divide(denom, np.float32(math.sqrt(1 - b2**step)), dtype=np.float32)
    denom = np.add(denom, np.float32(eps), dtype=np.float32)
    weight = np.asarray(
        weight.astype(np.float64) - lr / (1 - b1**step) * mean.astype(np.float64) / denom, dtype=np.float32
    )
    rounded = tuple(
        reference_round(values, rounding_key(name, step, role, seed))
        for values, role in zip((weight, mean, variance), ("weight", "exp_avg", "exp_avg_sq"), strict=False)
    )
    return (rounded, (weight, mean, variance)) if return_unrounded else rounded


def test_bf16_state_and_rounding_match_independent_cpu_reference():
    checks = {}
    # Positive/negative values, signed zero, BF16 subnormal and exponent boundaries.
    raw = np.array(
        [
            0,
            0x80000000,
            0x3F800001,
            0x3F807FFF,
            0x3F808000,
            0xBF808000,
            0xBF80FFFF,
            1,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x7F000001,
            0xFF000001,
        ],
        dtype=np.uint32,
    )
    x = torch.from_numpy(raw.view(np.float32).copy())
    key = rounding_key("probe", 2, "weight", 78)
    observed = stochastic_bf16(x, key=key).float().numpy()
    assert np.array_equal(observed.view(np.uint32), reference_round(x.numpy(), key).view(np.uint32))
    uniform = counter_random16(257, key=key, offset=4001, device="cpu").numpy()
    assert np.array_equal(uniform, reference_uniform16(key, 4001, 257))
    checks["scalar_bit_reference"] = True
    values = torch.linspace(-2, 3, 11003)
    whole = stochastic_bf16(values, key=key)
    pieces = torch.cat(
        [
            stochastic_bf16(values[start:end].contiguous(), key=key, offset=start)
            for start, end in ((0, 511), (511, 8001), (8001, 11003))
        ]
    )
    assert torch.equal(whole, pieces)
    checks["chunk_and_uneven_shard_invariance"] = True
    streams = [
        counter_random16(5000, key=rounding_key("probe", 2, role, 78), offset=0, device="cpu")
        for role in ("weight", "exp_avg", "exp_avg_sq")
    ]
    assert all(not torch.equal(streams[i], streams[j]) for i in range(3) for j in range(i))
    checks["independent_named_state_streams"] = True
    # A BF16 RNE update below half an ULP freezes exactly; SR must still update.
    n = 1_000_000
    original = torch.full((n,), 0.02, dtype=torch.bfloat16)
    updated = original.float() - 1e-6
    rne = updated.to(torch.bfloat16)
    sr = stochastic_bf16(updated, key=rounding_key("small_update", 1, "weight", 5))
    changed = int((sr != original).sum())
    assert int((rne != original).sum()) == 0 and changed > 1000
    error = (sr.double() - updated.double()).mean().item()
    standard_error = (sr.double() - updated.double()).std().item() / math.sqrt(n)
    assert abs(error) <= 6 * standard_error + 1e-12
    checks["small_update"] = dict(
        elements=n,
        rne_changed=0,
        stochastic_changed=changed,
        mean_rounding_error=error,
        standard_error=standard_error,
        within_six_sigma=True,
    )
    for invalid in (float("inf"), float("nan")):
        try:
            stochastic_bf16(torch.tensor([invalid]), key=key)
        except FloatingPointError:
            pass
        else:
            raise AssertionError("Nonfinite rounding must fail")
    checks["nonfinite_fail_closed"] = True
    name = "cpu_reference.policy"
    seed = 20261106
    parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.7, 6001).to(torch.bfloat16))
    parameter._qwen36_parameter_name = name
    state = CpuBf16Moments(cpu_budget_bytes=parameter.numel() * 4)
    gradients = GradientStore(retained_bytes=0, retained_device="cpu")
    optimizer = Bf16CpuAdamW(
        [parameter],
        gradients=gradients,
        moments=state,
        lr=3e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        chunk_bytes=1028,
        rounding_seed=seed,
    )
    initial = parameter.detach().float().numpy().copy()
    reference = (initial, np.zeros_like(initial), np.zeros_like(initial))
    comparisons = []
    for step in range(1, 4):
        g = torch.sin(torch.arange(parameter.numel(), dtype=torch.float32) * 0.01 + step)
        # Exercise real FP32 capture accumulation and CPU spill, independently of FSDP hooks.
        gradients.capture_reduced(parameter, g * 0.3)
        gradients.capture_reduced(parameter, g * 0.7)
        actual_grad = gradients.slots[parameter].gradient.numpy().copy()
        norm = gradients.clip_grad_norm(
            100, expected_local_numel=parameter.numel(), expected_global_numel=parameter.numel()
        )
        assert abs(norm - np.linalg.norm(actual_grad.astype(np.float64))) < 1e-9
        reference = numpy_adam(
            *reference, actual_grad, step=step, lr=3e-5, betas=(0.9, 0.999), eps=1e-8, wd=0.01, name=name, seed=seed
        )
        optimizer.step()
        handles = optimizer.state[parameter]["moments"]
        outputs = (
            parameter.detach().float().numpy(),
            handles["first"].float().numpy(),
            handles["second"].float().numpy(),
        )
        mismatches = [
            int(np.count_nonzero(out.view(np.uint32) != ref.view(np.uint32)))
            for out, ref in zip(outputs, reference, strict=False)
        ]
        # Independent NumPy arithmetic can differ by FP32 FMA rounding before SR;
        # enforce no BF16 differences for this fixed dataset and seed.
        assert mismatches == [0, 0, 0], mismatches
        comparisons.append(
            dict(
                step=step,
                mismatches_weight_mean_variance=mismatches,
                changed_weight_elements=optimizer.changed_weight_elements,
            )
        )
        optimizer.zero_grad()
    assert all(
        t.dtype == torch.bfloat16 and t.device.type == "cpu"
        for item in state.handles
        for t in (item["first"], item["second"])
    )
    assert state.bytes == parameter.numel() * 4 and len(state.handles) == 1
    assert all(
        not isinstance(item, torch.Tensor) or item.dtype != torch.float32
        for item in optimizer.state[parameter].values()
    )
    checks["three_step_independent_adam_reference"] = comparisons
    checks["persistent_storage"] = optimizer.audit()
    try:
        CpuBf16Moments(cpu_budget_bytes=3).plan([("x", parameter)])
    except MemoryError:
        pass
    else:
        raise AssertionError("Overbudget state must not use disk")
    checks["strict_memory_budget_without_disk_fallback"] = True
