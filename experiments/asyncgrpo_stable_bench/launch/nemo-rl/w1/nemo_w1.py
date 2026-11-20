"""NeMo-RL's own examples/run_grpo.py (sync or async GRPO), with one addition: the `w1_math` env (nemo_w1_env.py,
the bench's shared reward) is registered before run_grpo builds its environments. No NeMo-RL file is changed.

    python nemo_w1.py --config <yaml> [hydra overrides]
"""

import runpy
import sys

from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.utils import register_env

FQN = "nemo_w1_env.W1MathEnv"
register_env("w1_math", FQN)
ACTOR_ENVIRONMENT_REGISTRY[FQN] = PY_EXECUTABLES.SYSTEM

if __name__ == "__main__":
    sys.argv[0] = "/opt/nemo-rl/examples/run_grpo.py"
    runpy.run_path(sys.argv[0], run_name="__main__")
