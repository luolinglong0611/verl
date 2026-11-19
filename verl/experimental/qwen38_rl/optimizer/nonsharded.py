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
"""Experimental MCore optimizer wrapper preserving local model allocations.

Used only by the opt-in factory adapter. It deliberately does not support
checkpoint persistence; production recovery must be designed separately.
"""

import torch
from megatron.core.optimizer.optimizer import MegatronOptimizer

from verl.utils.device import get_device_name


class NonShardedBf16Optimizer(MegatronOptimizer):
    def __init__(self, optimizer, config, grad_scaler=None, init_state_fn=None):
        if grad_scaler is not None:
            raise ValueError("BF16 candidate does not support a loss scaler")
        super().__init__(optimizer, config, init_state_fn)
        self.is_stub_optimizer = optimizer is None

    def zero_grad(self, set_to_none=True):
        if self.is_stub_optimizer:
            return
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for p in self.get_parameters():
            p.decoupled_grad = None

    def prepare_grads(self):
        if not self.is_stub_optimizer:
            for p in self.get_parameters():
                grad = getattr(p, "main_grad", None)
                if grad is None or grad.dtype != torch.bfloat16 or grad.shape != p.shape:
                    raise ValueError("actual Megatron BF16 main_grad missing")
                p.decoupled_grad = grad
        return False

    def step_with_ready_grads(self):
        if not self.is_stub_optimizer:
            self.optimizer.step()
        return True

    def get_loss_scale(self):
        return torch.ones((), dtype=torch.float32, device=get_device_name())

    def reload_model_params(self, state_dict=None):
        if state_dict is not None:
            raise NotImplementedError("checkpoint reload not qualified")

    def state_dict(self):
        raise NotImplementedError("candidate checkpointing not qualified")

    def load_state_dict(self, state_dict):
        raise NotImplementedError("candidate checkpointing not qualified")

    def sharded_state_dict(self, *args, **kwargs):
        raise NotImplementedError("candidate checkpointing not qualified")

    def step(self):
        raise RuntimeError("call the guarded chained optimizer, not an inner part")
