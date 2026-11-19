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
"""Bounded, nonmutating BF16 CUDA finite check; caller performs all-rank vote."""

import torch

from verl.utils.device import get_device_id, get_device_name


def batched_finite(gradients, max_views=32, view_numel=1048576):
    if type(max_views) is not int or type(view_numel) is not int or min(max_views, view_numel) < 1:
        raise ValueError("invalid finite-check batch limits")
    device = torch.device(get_device_name(), get_device_id())
    finite = torch.ones((), dtype=torch.int32, device=device)
    batch = []

    def flush():
        if batch:
            norms = torch._foreach_norm(batch, float("inf"), dtype=torch.float32)
            finite.bitwise_and_(torch.isfinite(torch.stack(norms)).all().to(torch.int32))
            batch.clear()

    for grad in gradients:
        if grad is None:
            continue
        if grad.device != device or grad.dtype != torch.bfloat16 or not grad.is_contiguous():
            raise ValueError("finite-check input must be a contiguous local CUDA BF16 shard")
        flat = grad.reshape(-1)
        for start in range(0, flat.numel(), view_numel):
            batch.append(flat[start : start + view_numel])
            if len(batch) == max_views:
                flush()
    flush()
    return finite
