# W1-S (math, Qwen3.5-2B) as `openrlhf.cli.train_ppo_ray` flags. Sourced by launch/openrlhf/w1.sbatch, which sets
# MODEL, DATA, RUN_DIR, NGPU_INF, NGPU_TRAIN. Knob-by-knob mapping and semantics: EQUIVALENCE.md.
W1_ARGS=(
   # model: bf16 params, DeepSpeed keeps an fp32 master copy in the optimizer; text-only, vision tower frozen
   --actor.model_name_or_path "$MODEL"
   --actor.freeze_visual_encoder
   --ds.param_dtype bf16
   --ds.zero_stage 2
   --actor.gradient_checkpointing_enable
   --ds.attn_implementation flash_attention_2
   # no --ds.packing_samples / --train.dynamic_batch_enable: packing leaks gated-DeltaNet state across samples
   # (envs/openrlhf/NOTES.md item 8); micro-batch 1 means no padding either

   # placement: 4 vLLM engines (TP 1) + 4 DeepSpeed ranks, not colocated; async with queue 1 (max staleness 1)
   --vllm.num_engines "$NGPU_INF"
   --vllm.tensor_parallel_size 1
   --vllm.gpu_memory_utilization 0.9
   --vllm.sync_backend nccl
   # no --vllm.enforce_eager: the CLI default (CUDA graphs on). The example scripts pass it for colocated sleep mode; in
   # eager mode the 4 engines sat at ~31% SM util and one 256-rollout batch took ~25 min (NOTES.md item 17)
   --vllm.enable_prefix_caching
   --actor.num_nodes 1
   --actor.num_gpus_per_node "$NGPU_TRAIN"
   --train.async_enable
   --train.async_queue_size 1

   # data: the shared prompt, pre-rendered with the model's chat template by w1_data.py (OpenRLHF can't pass template
   # kwargs such as enable_thinking), so no --data.apply_chat_template; THINKING picks the file in w1.sbatch
   --data.prompt_dataset "$DATA"
   --data.input_key prompt
   --data.label_key label
   --reward.remote_url "$BENCH/configs/openrlhf/w1_reward.py"

   # batch: 32 prompts x 8 = 256 rollouts per step, one optimizer step per rollout batch
   --rollout.batch_size 32
   --rollout.n_samples_per_prompt 8
   --train.batch_size 256
   --train.micro_batch_size 1
   --rollout.micro_batch_size 1
   --train.max_epochs 1
   --train.num_episodes 1
   --data.max_samples $((32 * ${MAX_STEPS:-200}))

   # lengths (rev. 2): max_len is prompt + completion; a prompt is left-truncated only beyond 18432 - 16384 = 2048 tokens
   --rollout.max_new_tokens 16384
   --data.max_len 18432
   --rollout.temperature 1.0
   --rollout.top_p 1.0

   # optimizer: AdamW 1e-6 constant, no warmup, wd 0, betas (0.9, 0.999), clip 1.0
   --actor.adam.lr 1e-6
   --actor.adam.betas 0.9 0.999
   --actor.adam.weight_decay 0.0
   --actor.lr_scheduler constant
   --actor.lr_warmup_ratio 0
   --actor.max_norm 1.0

   # loss: Dr. GRPO advantage (group mean, no std, no batch whitening), PPO clip 0.2/0.28, global token mean,
   # no KL (init_coef 0 => no reference model), no entropy term; token-level IS mask against vLLM logprobs ([0.5, 5])
   --algo.advantage.estimator dr_grpo
   --actor.eps_clip_low_high 0.2 0.28
   --actor.loss_agg_mode token-mean
   --algo.kl.init_coef 0
   --algo.advantage.is_correction_level token
   --algo.advantage.is_correction_mode mask
   --algo.advantage.is_correction_threshold 0.5 5.0

   --train.seed 42
   --ckpt.save_steps 50
   --ckpt.save_hf
   --ckpt.max_num 1
   --ckpt.load_enable
   --ckpt.path "$RUN_DIR/ckpt"
   --ckpt.output_dir "$RUN_DIR/final"
   --logger.tensorboard_dir "$RUN_DIR/tb"
   --logger.logging_steps 1
   --eval.steps -1
)
