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

import sys
from types import ModuleType

import pytest

from verl.workers.rollout.base import get_rollout_class
from verl.workers.rollout.replica import get_rollout_replica_class


@pytest.mark.parametrize(
    "world72, module_name, adapter_name, replica_name",
    [
        (False, "rtp_verl_backend", "RTPServerAdapter", "RTPReplica"),
        (True, "qwen38_verl_backend", "World72ServerAdapter", "World72Replica"),
    ],
)
def test_rtp_adapter_and_replica_select_same_profile(monkeypatch, world72, module_name, adapter_name, replica_name):
    monkeypatch.setenv("RTP_LLM_RL_WORLD72", "1" if world72 else "0")
    module = ModuleType(module_name)
    adapter = type(adapter_name, (), {})
    replica = type(replica_name, (), {})
    setattr(module, adapter_name, adapter)
    setattr(module, replica_name, replica)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert get_rollout_class("rtp") is adapter
    assert get_rollout_replica_class("rtp") is replica


def test_missing_rtp_reports_required_adapter(monkeypatch):
    monkeypatch.delenv("RTP_LLM_RL_WORLD72", raising=False)
    monkeypatch.setitem(sys.modules, "rtp_verl_backend", None)
    with pytest.raises(ImportError, match="example/rl/qwen_moe_rl"):
        get_rollout_class("rtp")
