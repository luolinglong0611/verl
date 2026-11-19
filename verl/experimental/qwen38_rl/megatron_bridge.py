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
"""Megatron Bridge for the rank-local BF16 Qwen3.8 checkpoint importer."""

import json
import os
from pathlib import Path

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.utils import unwrap_model
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge
from transformers import AutoConfig

from .megatron_loader import TRAIN_TP, TRAIN_VOCAB_DIVISOR, _checkpoint_from_env, _world_gate, load_policy
from .remote_checkpoint import Metadata


class Q38TextModelBridge(Qwen35MoEBridge):
    def mapping_registry(self):
        return MegatronMappingRegistry(*self._get_moe_lm_mappings(experts_packed=True))


class Q38TextAutoBridge(AutoBridge):
    @classmethod
    def from_hf_pretrained(cls, path, **kwargs):
        if set(kwargs) - {"trust_remote_code", "local_files_only"}:
            raise ValueError("unsupported Qwen3.8 Bridge source option")
        metadata = Metadata.from_files(path, os.environ["Q38_OBJECT_MANIFEST"], os.environ["Q38_POLICY_MANIFEST"])
        cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if cfg.architectures != metadata.config["architectures"] or cfg.model_type != metadata.config["model_type"]:
            raise ValueError("Transformers resolved a different Qwen3.8 architecture")
        # Model construction uses the main policy only.
        cfg.mtp_num_hidden_layers = 0
        bridge = cls.from_hf_config(cfg)
        bridge._q38_local_path = Path(path).resolve()
        return bridge

    @property
    def _model_bridge(self):
        bridge = Q38TextModelBridge()
        bridge.export_weight_dtype = self.export_weight_dtype
        return bridge

    def to_megatron_provider(self, load_weights=True, hf_path=None):
        if load_weights or hf_path is not None:
            raise ValueError("Qwen3.8 requires the bounded post-construction checkpoint import")
        provider = super().to_megatron_provider(load_weights=False)
        if provider.vocab_size % (TRAIN_VOCAB_DIVISOR * TRAIN_TP):
            raise ValueError("Qwen3.8 vocabulary does not divide exactly across TP8")
        provider.make_vocab_size_divisible_by = TRAIN_VOCAB_DIVISOR
        provider.mtp_num_layers = 0
        provider.perform_initialization = False
        return provider

    def load_hf_weights(self, model, hf_path=None, allowed_mismatched_params=None):
        path = self._q38_local_path
        error = None
        try:
            if allowed_mismatched_params or (hf_path is not None and Path(hf_path).resolve() != path):
                raise ValueError("invalid Qwen3.8 actor checkpoint import request")
            raw = unwrap_model(model)
            if len(raw) != 1:
                raise ValueError("Qwen3.8 TP8/EP64 actor requires one local PP1 model")
            checkpoint = _checkpoint_from_env(path)
        except Exception as exc:
            error = exc
        _world_gate(error, "pinned actor checkpoint metadata")
        receipt = load_policy(self._model_bridge, self.hf_pretrained, raw[0], checkpoint)
        self.unquantized_state_dict = None
        print("Q38_TRAIN_CHECKPOINT " + json.dumps(receipt, sort_keys=True), flush=True)
        return model
