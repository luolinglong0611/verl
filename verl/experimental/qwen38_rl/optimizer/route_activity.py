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
"""Track expert activity from pre-padding dispatched token counts.

Only discrete weightN/biasN grouped expert parameters are supported. The
module output must participate in backward before its counts are committed.
"""

import re

import torch

from verl.utils.device import get_device_name, get_torch_device, is_cuda_available


class RoutedActivity:
    def __init__(self, routed_module, linears):
        self.module = routed_module
        self.n = int(linears[0].num_gemms)
        self.physical_n = int(getattr(routed_module, "fixed_physical_slots", self.n))
        if self.physical_n < self.n or int(getattr(routed_module, "fixed_real_experts", self.n)) != self.n:
            raise ValueError("fixed-slot expert metadata disagrees with grouped parameters")
        self.params = {}
        for linear in linears:
            if int(linear.num_gemms) != self.n or getattr(linear, "single_grouped_weight", False):
                raise ValueError("grouped weights without per-expert parameters are unsupported")
            present = set()
            for name, param in linear.named_parameters(recurse=False):
                match = re.fullmatch(r"(weight|bias)(\d+)", name)
                if match is None:
                    raise ValueError(f"unsupported grouped expert parameter {name}")
                idx = int(match.group(2))
                if idx >= self.n:
                    raise ValueError("expert index exceeds dispatch vector")
                self.params[id(param)] = (param, idx)
                if match.group(1) == "weight":
                    present.add(idx)
            if present != set(range(self.n)):
                raise ValueError("missing discrete expert weight")
        self.device = next(iter(self.params.values()))[0].device
        if not is_cuda_available or self.device.type != get_device_name():
            raise ValueError("CUDA routed expert parameters required")
        self.active = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.masks = []
        self.handle = routed_module.register_forward_hook(self._observe, with_kwargs=True)

    def _observe(self, module, args, kwargs, output):
        if not torch.is_grad_enabled():
            return
        counts = args[1] if len(args) > 1 else kwargs.get("tokens_per_expert", kwargs.get("m_splits"))
        if counts is None:
            raise ValueError("routed module did not provide expert token counts")
        counts = torch.as_tensor(counts, device=self.device)
        if counts.shape != (self.physical_n,) or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("unexpected expert token-count vector")
        if self.physical_n > self.n and bool((counts[self.n :] != 0).any()):
            raise RuntimeError("dummy expert received a token")
        selected = counts[: self.n] > 0
        ready = get_torch_device().Event()
        ready.record(get_torch_device().current_stream(self.device))
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
            return

        def mark(grad):
            # Collect independent masks here; merge on one stream after all
            # backward work so concurrent backward streams cannot race ORs.
            self.masks.append((selected, ready))
            return grad

        tensor.register_hook(mark)

    def synchronize(self):
        stream = get_torch_device().current_stream(self.device)
        for mask, event in self.masks:
            stream.wait_event(event)
            self.active.logical_or_(mask)
        self.masks.clear()

    def reset(self):
        self.synchronize()
        self.active.zero_()

    def flags(self):
        self.synchronize()
        active = self.active.cpu().tolist()
        return {marker: active[idx] for marker, (_, idx) in self.params.items()}
