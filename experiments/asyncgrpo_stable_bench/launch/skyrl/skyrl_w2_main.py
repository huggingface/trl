"""W2 entrypoint: SkyRL's fully async Harbor trainer (examples/train_integrations/harbor/entrypoints/main_harbor_fully_async.py)
with two W2 changes, both recorded in configs/skyrl/EQUIVALENCE.md (W2):

1. Budget-hit episodes are trained with their verifier score (SHARED_W2 "failed episodes"). Stock SkyRL treats Harbor's
   AgentTimeoutError as a failure: no retry, and the whole group is loss-masked. With ~40-50% of 9B episodes hitting the
   600 s budget, almost every group of 8 would be zeroed. Harbor still runs the verifier after an agent timeout and
   terminus-2 keeps the rollout details of every finished call, so the episode is used as-is. Everything else (retry on a
   missing verifier result, context overflow -> reward 0, one failed episode zeroes its group) is SkyRL's code unchanged.
2. W2 token counts per step, logged as `w2/*` scalars: `trained_tokens` (SHARED_W2: every per-call sample's prompt +
   completion), `loss_tokens`, `forwarded_tokens` (prompt + response of each sequence after SkyRL's prefix-aware merge,
   which is what the FSDP workers run, one unpadded sequence per micro-batch), the sample counts before and after merging.

3. The shared-arm loss `ppo_rollout`: one-sided PPO-clip against the sampler's (vLLM) logprobs, registered through SkyRL's
   public PolicyLossRegistry (no source patch). Copied from main's W1 fix
   (/fsx/amine_dirhoussi/stable-bench-runs/skyrl/collapse-364970/collapse_main.py, envs/skyrl/COLLAPSE_364970.md), which
   replaces the two-sided `rollout_is` mask that let W1-S 364970 drift and collapse.

The Harbor trial config comes from one YAML file (`--harbor-config`), not SkyRL's default.yaml merged with overrides, so
nothing Daytona-specific leaks in.
"""

import sys
import time
from copy import deepcopy
from uuid import uuid4

import ray
import torch
import yaml
from examples.train_integrations.harbor import harbor_generator as hg
from examples.train_integrations.harbor.entrypoints.main_harbor import HarborExp, HarborSkyRLConfig, _deep_merge
from examples.train_integrations.harbor.trial_metrics import TrialAttempts
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import off_policy_correction_enabled
from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry, reduce_loss
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean, safe_exp_delta
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


def ppo_rollout_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    ratio = safe_exp_delta(log_probs - rollout_logprobs, clip=20.0, out_dtype=log_probs.dtype)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    clip_ratio = masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
    return reduce_loss(loss, loss_mask), {"clip_ratio": clip_ratio}


PolicyLossRegistry.register("ppo_rollout", ppo_rollout_policy_loss)


class W2HarborGenerator(hg.HarborGenerator):
    _logged_prompt = False

    async def _harbor_agent_loop(self, prompt, trajectory_id, cache_salt=None, attempts=None):
        """SkyRL's `_harbor_agent_loop` with one change, marked W2: an AgentTimeoutError with a verifier result is scored."""
        agent_loop_start_time = time.monotonic()
        attempts = attempts if attempts is not None else TrialAttempts()
        reward = None
        results = None
        rollout_details = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False

        for i in range(hg.MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i + 1}/{hg.MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            session_id = uuid4().hex
            attempts.start()
            try:
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = session_id
                if cache_salt is not None:
                    llm_kwargs = config["agent"]["kwargs"].setdefault("llm_kwargs", {})
                    extra_body = llm_kwargs.setdefault("extra_body", {})
                    extra_body["cache_salt"] = cache_salt
                trial_config = TrialConfig.model_validate(config)
                trial = await Trial.create(trial_config)

                async with self._rate_limiter:
                    results = await trial.run()
                attempts.record(results)

                exc_type = results.exception_info.exception_type if results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                if is_agent_timeout_error and not results.verifier_result:
                    # W2: a budget hit the verifier never scored is unscored, as in stock SkyRL (no retry, group masked)
                    logger.debug(f"{prefix} hit AgentTimeoutError without a verifier result (no retry).")
                    break
                elif is_context_length_error:
                    logger.debug(f"{prefix} hit ContextLengthExceededError, setting reward=0.")
                    reward = 0.0
                elif not results.verifier_result:
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}.")
                    continue
                else:
                    # W2: also the AgentTimeoutError case: verified as-is, trained with that score
                    reward = float(results.verifier_result.rewards["reward"])

                rollout_details = results.agent_result.rollout_details
                num_turns = results.agent_result.metadata["n_episodes"]

                if (
                    rollout_details
                    and len(rollout_details) >= 1
                    and len(rollout_details[0].get("completion_token_ids", [])) > 0
                ):
                    successful = True
                    is_agent_timeout_error = False
                    if not W2HarborGenerator._logged_prompt:
                        W2HarborGenerator._logged_prompt = True
                        tail = self.tokenizer.decode(rollout_details[0]["prompt_token_ids"][0][-24:])
                        logger.info(f"W2 first-call prompt tail (thinking-off check): {tail!r}")
                    break
                else:
                    logger.warning(f"{prefix} failed: empty/missing rollout_details.")
            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}.")
                attempts.fail(e)
                continue
            finally:
                await self.inference_engine_client.finish_session(session_id)

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            logger.warning(
                f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), will set loss mask to [0]."
            )
            return hg.HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=None,
                stop_reason=stop_reason,
                e2e_time=time.monotonic() - agent_loop_start_time,
            )
        return hg.HarborTrajectoryOutput(
            trajectory_id=trajectory_id,
            rollout_details=rollout_details,
            reward=reward,
            num_turns=num_turns,
            stop_reason="context_length" if is_context_length_error else "complete",
            e2e_time=time.monotonic() - agent_loop_start_time,
        )


def _seq_tokens(out):
    """(samples, prompt + response tokens, loss tokens) over the real samples (not the [0] placeholders of masked groups)."""
    n = tokens = loss = 0
    for p, r, m in zip(out["prompt_token_ids"], out["response_ids"], out["loss_masks"], strict=False):
        if len(p) == 1 and len(r) == 1 and sum(m) == 0:
            continue
        n += 1
        tokens += len(p) + len(r)
        loss += sum(m)
    return n, tokens, loss


class W2Trainer(FullyAsyncRayPPOTrainer):
    def _skip_policy_forward(self, training_input):
        # `ppo_rollout` reads only the rollout logprobs, like SkyRL's rollout_is / dppo, but isn't in its
        # LOSSES_WITHOUT_OLD_LOGPROBS, so the trainer would run a no-grad policy forward over every sample for old logprobs
        # nobody reads. Skip it under the same conditions SkyRL uses for its own rollout-anchored losses.
        algorithm = self.cfg.trainer.algorithm
        if algorithm.policy_loss_type == "ppo_rollout":
            return (
                training_input.get("rollout_logprobs", None) is not None
                and not algorithm.use_kl_in_reward
                and not off_policy_correction_enabled(algorithm.off_policy_correction)
            )
        return super()._skip_policy_forward(training_input)

    def postprocess_generator_output(self, generator_output, uids, **kwargs):
        n_before, trained, loss = _seq_tokens(generator_output)
        generator_output, uids = super().postprocess_generator_output(generator_output, uids, **kwargs)
        n_after, forwarded, _ = _seq_tokens(generator_output)
        self.all_metrics.update(
            {
                "w2/trained_tokens": trained,
                "w2/loss_tokens": loss,
                "w2/forwarded_tokens": forwarded,
                "w2/prefix_share_ratio": trained / forwarded if forwarded else 0.0,
                "w2/samples_before_merge": n_before,
                "w2/samples_after_merge": n_after,
                "w2/max_seq_tokens": max(
                    (
                        len(p) + len(r)
                        for p, r in zip(
                            generator_output["prompt_token_ids"], generator_output["response_ids"], strict=False
                        )
                    ),
                    default=0,
                ),
            }
        )
        return generator_output, uids


class W2Exp(HarborExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return W2HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_trainer(
        self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg
    ):
        return W2Trainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    W2Exp(cfg).run()


def main() -> None:
    argv = sys.argv[1:]
    harbor_file = next(a.split("=", 1)[1] for a in argv if a.startswith("--harbor-config="))
    cfg = HarborSkyRLConfig.from_cli_overrides([a for a in argv if not a.startswith("--harbor-config=")])
    with open(harbor_file) as f:
        base = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(base, cfg.harbor_trial_config)
    validate_cfg(cfg)
    assert cfg.trainer.algorithm.max_seq_len is not None, (
        "trainer.algorithm.max_seq_len must be set for Harbor training"
    )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
