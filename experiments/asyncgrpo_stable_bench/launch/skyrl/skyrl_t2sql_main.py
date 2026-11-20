"""Text-to-SQL entrypoint: SkyRL's standard (synchronous, colocated) trainer, skyrl.train.entrypoints.main_base, plus
the t2sql env (launch/skyrl/skyrl_t2sql_env.py) registered in the training process."""

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    register(id="t2sql", entry_point="skyrl_t2sql_env:Text2SQLEnv")
    BasePPOExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
