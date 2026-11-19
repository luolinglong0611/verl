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
from types import ModuleType, SimpleNamespace

import pytest
import torch

from examples.rl_rollout.qwen38_grpo_batch import exact_token_rows


@pytest.mark.parametrize("change_logprob", [False, True])
def test_native_tokens_and_fp32_scores_survive_verl_batching(monkeypatch, change_logprob):
    data_module = ModuleType("rllm.trainer.verl.dataclass")
    transform_module = ModuleType("rllm.trainer.verl.transform")

    def new_data():
        return SimpleNamespace(prompts=[], responses=[], traj_mask=[], rollout_logprobs=[], repeat_counts=[])

    def process(episode, _task_id, data):
        step = episode.trajectories[0].steps[0].model_output
        data.prompts.append(torch.tensor(step.prompt_ids))
        data.responses.append(torch.tensor([step.completion_ids[0], 37, step.completion_ids[1]]))
        data.traj_mask.append(torch.tensor([True, False, True]))
        data.rollout_logprobs.append(
            torch.tensor([step.logprobs[0] if not change_logprob else 0.0, 0.0, step.logprobs[1]], dtype=torch.float32)
        )
        return 1

    def build(data, pad_id, max_prompt, max_response, _):
        assert (pad_id, max_prompt, max_response) == (0, 2, 3)
        return SimpleNamespace(
            batch={
                "response_mask": torch.stack(data.traj_mask),
                "rollout_log_probs": torch.stack(data.rollout_logprobs),
            }
        )

    data_module.AccumulatedData = new_data
    transform_module._process_episode = process
    transform_module._batch_tensors_and_build_data_proto = build
    monkeypatch.setitem(sys.modules, data_module.__name__, data_module)
    monkeypatch.setitem(sys.modules, transform_module.__name__, transform_module)
    output = SimpleNamespace(prompt_ids=[13, 41], completion_ids=[7, 0], logprobs=[-0.125, -0.25])
    step = SimpleNamespace(model_output=output)
    episode = SimpleNamespace(task_id="sample", trajectories=[SimpleNamespace(steps=[step])])
    engine = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

    if change_logprob:
        with pytest.raises(ValueError, match="changed an original FP32 logprob"):
            exact_token_rows([episode], engine)
        return

    batch = exact_token_rows([episode], engine).batch
    # ID 0 is a sampled stop token. The tool result is visible to attention but excluded from PPO loss.
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert batch["response_mask"].tolist() == [[True, False, True]]
    scores = batch["rollout_log_probs"][batch["response_mask"]]
    assert torch.equal(scores.view(torch.int32), torch.tensor(output.logprobs, dtype=torch.float32).view(torch.int32))
