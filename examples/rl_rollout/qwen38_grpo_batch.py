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
"""Freeze DeepScaleR order and audit exact on-policy Megatron training rows."""

import hashlib
import json
import math
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATA_SEED = SAMPLING_SEED = 0
MAX_CONTEXT = 16384
PROMPTS = 512
ROLLOUTS = 8


FORCED_PYTHON_SUFFIX = (
    "\n\nUse the Python tool at least once to check a calculation before giving the final boxed answer."
)
LIMIT_TERMINATIONS = {"context_limit", "tool_limit"}
FILTERED_GRPO = "q38_filtered_grpo"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def original_row(row):
    original = row["original_question"]
    if row["question"] != original + FORCED_PYTHON_SUFFIX:
        raise ValueError("unexpected transformation of the pinned DeepScaleR question")
    normalized = " ".join(unicodedata.normalize("NFKC", original).split())
    if not normalized or not row["ground_truth"]:
        raise ValueError("DeepScaleR contains an empty question or answer")
    return {**row, "question": original, "id": f"deepscaler-train-{row['source_row_index']}"}


def batch_lock(batch, dataset_lock):
    if len(batch) != PROMPTS or len({row["source_row_index"] for row in batch}) != PROMPTS:
        raise ValueError("the normal dataloader did not yield the requested distinct first batch")
    clean = [original_row(row) for row in batch]
    if len({" ".join(unicodedata.normalize("NFKC", row["question"]).split()) for row in clean}) != PROMPTS:
        raise ValueError("the first batch contains duplicate DeepScaleR problems")
    manifest = {
        "schema": "q38-first-deepscaler-training-batch-v1",
        "step": 1,
        "dataset": dataset_lock,
        "data_seed": DATA_SEED,
        "dataloader_epoch": 0,
        "dataloader_cursor_before_batch": 0,
        "rollouts_per_prompt": ROLLOUTS,
        "sampling_seed": SAMPLING_SEED,
        "source_row_indices": [row["source_row_index"] for row in clean],
        "processed_batch_sha256": sha(b"".join(canonical(row) for row in clean)),
        "original_batch_sha256": sha(b"".join(canonical(row) for row in batch)),
        "rows": [
            {
                "position": position,
                "source_row_index": row["source_row_index"],
                "task_id": row["id"],
                "question_sha256": sha(row["question"].encode()),
                "answer_sha256": sha(row["ground_truth"].encode()),
            }
            for position, row in enumerate(clean)
        ],
    }
    return clean, manifest


def prepare_batch(loader, dataset_lock, output):
    if loader.state_dict() != {"epoch": 0, "cursor": 0, "seed": DATA_SEED}:
        raise ValueError("a comparative run must start at the original DeepScaleR dataloader position")
    _, manifest = batch_lock(next(iter(loader.clone())), dataset_lock)
    data = canonical(manifest)
    replay = os.environ.get("Q38_FROZEN_BATCH_REPLAY")
    locked_batch = Path(__file__).with_name("FROZEN-BATCH.json")
    if locked_batch.is_file() and locked_batch.read_bytes() != data:
        raise ValueError("the training dataloader differs from the staged first-batch lock")
    if replay:
        if Path(replay).read_bytes() != data:
            raise ValueError("the current dataloader order and hashes differ from the frozen batch")
    with (output / "FROZEN-BATCH.json").open("xb") as stream:
        stream.write(data)
    return manifest


def eligible(episode):
    artifacts = episode.artifacts
    return artifacts["agent_termination"] not in LIMIT_TERMINATIONS or bool(artifacts["answer"])


def eligibility(episodes):
    accepted = [episode for episode in episodes if eligible(episode)]
    rewards = defaultdict(set)
    exclusions = Counter(episode.artifacts["agent_termination"] for episode in episodes if not eligible(episode))
    tokens = 0
    for episode in accepted:
        if len(episode.trajectories) != 1:
            raise ValueError("expected one mathematical trajectory for each episode")
        trajectory = episode.trajectories[0]
        if trajectory.reward is None or not math.isfinite(trajectory.reward) or not trajectory.steps:
            raise ValueError("eligible episode lacks finite reward or trainable model turns")
        rewards[episode.task_id].add(trajectory.reward)
        for step in trajectory.steps:
            tokens += len(step.model_output.completion_ids)
    return accepted, {
        "raw_trajectories": len(episodes),
        "eligible_trajectories": len(accepted),
        "excluded_by_reason": dict(exclusions),
        "eligible_question_groups": len(rewards),
        "reward_varying_question_groups": sum(len(group) > 1 for group in rewards.values()),
        "eligible_sampled_tokens": tokens,
    }


def install_filtered_grpo():
    """Keep a valid singleton and give it zero centered advantage."""
    import numpy as np
    from rllm.trainer.algorithms.advantage import calculate_grpo_advantages, register_rllm_adv_estimator

    @register_rllm_adv_estimator(FILTERED_GRPO)
    def filtered(rewards, algorithm_config, **kwargs):
        advantages, returns = calculate_grpo_advantages(rewards, algorithm_config, **kwargs)

        def center_singletons(values):
            return tuple(
                np.zeros_like(value) if len(group) == 1 else value for group, value in zip(rewards, values, strict=True)
            )

        return center_singletons(advantages), center_singletons(returns)

    return filtered


def exact_token_rows(episodes, engine):
    """Use verl's original merge and reward transform, with measured pad widths and masks."""
    import torch
    from rllm.trainer.verl.dataclass import AccumulatedData
    from rllm.trainer.verl.transform import _batch_tensors_and_build_data_proto, _process_episode

    data = AccumulatedData()
    source_tokens = 0
    for episode in episodes:
        steps = episode.trajectories[0].steps
        for step in steps:
            inference = step.model_output
            prompt, sampled, scores = inference.prompt_ids, inference.completion_ids, inference.logprobs
            if not prompt or not sampled or len(prompt) + len(sampled) > MAX_CONTEXT:
                raise ValueError("a real RTP training turn is outside the 16K context")
            if scores is None or len(scores) != len(sampled) or not all(map(math.isfinite, scores)):
                raise ValueError("native FP32 scores and sampled tokens differ in an eligible trajectory")
            source_tokens += len(sampled)
        row_start = len(data.responses)
        data.repeat_counts.append(_process_episode(episode, episode.task_id, data))
        for row in range(row_start, len(data.responses)):
            if len(data.prompts[row]) + len(data.responses[row]) > MAX_CONTEXT:
                raise ValueError("prefix merge exceeded the native context")
        actual_ids = torch.cat(
            [data.responses[i][data.traj_mask[i].bool()] for i in range(row_start, len(data.responses))]
        )
        expected_ids = torch.tensor([token for step in steps for token in step.model_output.completion_ids])
        if not torch.equal(actual_ids, expected_ids):
            raise ValueError("the verl response mask selects different tokens than RTP sampled")
        actual_lp = torch.cat(
            [data.rollout_logprobs[i][data.traj_mask[i].bool()] for i in range(row_start, len(data.responses))]
        )
        expected_lp = torch.tensor(
            [value for step in steps for value in step.model_output.logprobs], dtype=torch.float32
        )
        if actual_lp.dtype != torch.float32 or not torch.equal(
            actual_lp.view(torch.int32), expected_lp.view(torch.int32)
        ):
            raise ValueError("the verl transform changed an original FP32 logprob")
    max_prompt = max(map(len, data.prompts))
    max_response = max(map(len, data.responses))
    batch = _batch_tensors_and_build_data_proto(data, engine.tokenizer.pad_token_id, max_prompt, max_response, None)
    b = batch.batch
    # Some official stop token IDs can be equal to pad_token_id. Token IDs
    # do not determine validity. Positions and the real per-row lengths do.
    p_lengths = torch.tensor([len(prompt) for prompt in data.prompts])
    r_lengths = torch.tensor([len(response) for response in data.responses])
    p_mask = torch.arange(max_prompt).unsqueeze(0) >= max_prompt - p_lengths.unsqueeze(1)
    r_mask = torch.arange(max_response).unsqueeze(0) < r_lengths.unsqueeze(1)
    b["attention_mask"] = torch.cat((p_mask, r_mask), dim=1).long()
    b["position_ids"] = (b["attention_mask"].cumsum(dim=1) - 1) * b["attention_mask"]
    if (
        int(b["response_mask"].sum()) != source_tokens
        or bool((b["response_mask"].bool() & ~r_mask).any())
        or bool((b["rollout_log_probs"][~b["response_mask"].bool()] != 0).any())
    ):
        raise ValueError("tool results, template tokens or padding entered the training loss")
    return batch
