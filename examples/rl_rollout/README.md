# RL Rollout adaptations

## RTP rollout

Set `actor_rollout_ref.rollout.name=rtp` to resolve the project’s rollout adapter and replica through the verl registries. Add the RTP-LLM Qwen MoE integration directories to `PYTHONPATH` in the driver and Ray workers. Set `RTP_LLM_RL_WORLD72=1` in both places to select the single-policy TP8/attention-DP9/EP72 adapter. The split 64+32 driver uses its own RTP refit coordinator. [RTP-LLM integration source](https://github.com/alibaba/rtp-llm/tree/b169761ea861dde495531932614553096ed46b43/example/rl/qwen_moe_rl)

## Guarded NIXL checkpoint transfers

Use the standalone RTP adapter that calls `confirm_native_ack` after all native ranks consume a tensor. Register the checkpoint engine with these overrides.

```text
actor_rollout_ref.rollout.checkpoint_engine.backend=rtp_nixl_guarded
actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.checkpoint_engine.guarded_nixl_checkpoint_engine
```

The engine sends complete HF tensors and validates their SHA-256 before yielding them. The receiver forwards the current bucket before waiting for the next one, so consumers can rendezvous on a source tensor. A failed transfer retains its buffers and descriptors for the deployment supervisor to release after peer teardown. [Managed NIXL experiment](https://github.com/alibaba/rtp-llm/tree/28b0575202/example/rl/qwen_moe_rl/disaggregated/managed_reference_20261111)

## Split Qwen3.8 Megatron actor

The bridge reads the pinned Qwen3.8-2.4T-A95B BF16 checkpoint through a bounded range plan. Bind the actor to TP8/DP8/EP64 with PP1 and expert TP1. Select these overrides when configuring the actor through verl.

```text
actor_rollout_ref.actor.megatron.bridge_class=verl.experimental.qwen38_rl.megatron_bridge.Q38TextAutoBridge
actor_rollout_ref.actor.optim.optimizer_factory=verl.experimental.qwen38_rl.optimizer.qwen38.Qwen38Bf16OptimizerFactory
```

Supply the validated metadata directory as the model path. Set `Q38_OBJECT_MANIFEST`, `Q38_POLICY_MANIFEST`, `Q38_CHECKPOINT_CATALOG`, `Q38_CHECKPOINT_CATALOG_SHA256`, and `Q38_CHECKPOINT_ENDPOINTS` for the current allocation. Run `python -m verl.experimental.qwen38_rl.training_relay --help` for the planner and range-server arguments. Use the bridge and optimizer settings above in place of the V5 driver’s legacy `install_for_verl` calls. [V5 split training and refit profile](https://github.com/alibaba/rtp-llm/blob/144d222d7cdfe814fec6f75d9322859ba2e70c53/example/rl/qwen_moe_rl/disaggregated/QWEN38_SPLIT.md)

The optimizer keeps BF16 trainable weights and BF16 CPU Adam moments. It computes bounded FP32 update tiles with coordinate-based stochastic rounding. A CPU Gloo vote commits the update after each rank completes its local CUDA writes. Use the V5 BF16, DDP, and optimizer offload settings.

## World72 FSDP2 actor

Add this example directory to the driver and worker `PYTHONPATH`. Keep the existing World72 recipe and select the fork’s training runtime with these settings.

```sh
export Q38_ENABLE_BF16_STATE_RUNTIME=1
export RTP_LLM_RL_WORLD72=1
export Q38_HF_RTP_REPORT_ONLY=1
export Q38_LOGPROB_REPORT_ONLY=1
```

```text
actor_rollout_ref.model.external_lib=world72_training.qwen38_bf16_training_runtime
actor_rollout_ref.actor.optim.optimizer_impl=world72_training.qwen38_bf16_training_runtime
actor_rollout_ref.actor.optim.optimizer=Bf16CpuAdamW
```

The runtime loads each FSDP2 shard directly from the verified snapshot. It uses FP32 gradient reduction and bounded retention, BF16 CPU Adam moments, and a leased canonical export for RTP refit. The source and allocator checks remain tied to the reviewed Torch 2.13 reducer and World72 layout. Use the recipe’s pinned runtime, source manifests, and memory budgets. [World72 recipe and constraints](https://github.com/alibaba/rtp-llm/blob/b169761ea861dde495531932614553096ed46b43/example/rl/qwen_moe_rl/WORLD72.md)

## 16K GRPO batches

`qwen38_grpo_batch.py` carries the rLLM-to-verl batch adaptation from the 16K run. It builds attention from sequence lengths, verifies sampled token IDs and FP32 logprob bits, and excludes tool results and padding from the loss. `install_filtered_grpo()` registers the estimator used after filtering incomplete trajectories.

CPU tests in this fork exercise registration, guarded checkpoint leases, concurrent HTTP range reads, BF16 rounding and Adam arithmetic, and token and mask preservation. The linked GPU results come from the original runtime checkouts.
