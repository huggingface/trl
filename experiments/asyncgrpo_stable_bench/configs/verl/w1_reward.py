"""verl custom reward function for W1: calls the shared workloads/math/reward.py unchanged."""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "workloads", "math"))
from reward_safe import compute_score as shared_compute_score  # noqa: E402  (SHARED_W1 rev. 7)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return shared_compute_score(solution_str, ground_truth)
