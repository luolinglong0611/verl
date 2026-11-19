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
"""Load the rollout adapters shipped with the RTP-LLM Qwen MoE RL integration."""

import os

from verl.utils.import_utils import load_class_from_fqn


def _load_rtp_class(standard: str, world72: str) -> type:
    if os.environ.get("RTP_LLM_RL_WORLD72") == "1":
        module, name = "qwen38_verl_backend", world72
    else:
        module, name = "rtp_verl_backend", standard
    try:
        return load_class_from_fqn(f"{module}.{name}", "RTP rollout class")
    except ImportError as exc:
        raise ImportError(
            "RTP rollout requires the RTP-LLM example/rl/qwen_moe_rl adapter and its dependencies "
            f"on PYTHONPATH in the driver and Ray workers. {exc}"
        ) from exc


def load_rtp_server_adapter() -> type:
    """Resolve the actor-side adapter for the selected RTP topology."""
    return _load_rtp_class("RTPServerAdapter", "World72ServerAdapter")


def load_rtp_replica() -> type:
    """Resolve the matching RTP replica, without importing RTP for other rollout engines."""
    return _load_rtp_class("RTPReplica", "World72Replica")
