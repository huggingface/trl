"""W1 entrypoint: SkyRL's fully async trainer (examples/train/fully_async/main_fully_async.py) plus the W1 math env."""

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register


class W1Exp(BasePPOExp):
    def get_trainer(
        self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg
    ):
        return FullyAsyncRayPPOTrainer(
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
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(id="w1_math", entry_point="skyrl_w1_env:W1MathEnv")
    W1Exp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
