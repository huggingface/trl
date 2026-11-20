"""`openenv ...` CLI with the agent's call budget passed to terminus-2 itself.

OpenEnv's terminus-2 seam has no `step_limit` mapping, so a rollout's `agent_step_limit` only reaches the capture
proxy, whose canned stop reply terminus-2 does not parse: it keeps asking until the agent timeout (envs/trl/
W2_STEP_TIME.md cause 4, ~31 extra calls and 120 s on the 28% of episodes that hit the budget). terminus-2 has a
native `max_turns` (one model call per turn with summarisation off), so this gives the seam a `step_limit` that sets
it, and the episode ends at call `agent_step_limit` and goes to the verifier.

    python harbor_serve.py harbor serve --dataset ... (same arguments as `openenv harbor serve`)

The patch is at module level so a child process that re-imports this file as `__mp_main__` gets it too.
"""

import dataclasses
import sys

from openenv.cli.__main__ import main
from openenv.harbor import seams


seams.SEAMS["terminus-2"] = dataclasses.replace(seams.SEAMS["terminus-2"], step_limit=lambda n: {"max_turns": n})

if __name__ == "__main__":
    sys.exit(main())
