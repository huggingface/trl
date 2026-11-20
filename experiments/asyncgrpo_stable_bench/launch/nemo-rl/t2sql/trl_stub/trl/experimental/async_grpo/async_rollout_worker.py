# Stub so t2sql_episode.py imports without TRL inside NeMo-RL's Ray workers (same as _install_trl_stub in nemo_t2sql.py).
from collections import namedtuple

AsyncRolloutLoop = object
Rollout = None
TurnRecord = namedtuple("TurnRecord", "prompt_ids output_ids logprobs")
