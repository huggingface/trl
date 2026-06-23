# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import torch

from trl.experimental.randopt import RandOptConfig, RandOptSearch, majority_vote


def test_randopt_run_returns_sorted_candidates_and_restores_parameters():
    model = torch.nn.Linear(4, 2, bias=False)
    initial_weight = model.weight.detach().clone()

    def score_fn(current_model: torch.nn.Module) -> float:
        weight_norm = current_model.weight.norm().item()
        # Higher score for smaller norm to make ranking deterministic enough.
        return -weight_norm

    config = RandOptConfig(population_size=8, sigma_values=[1e-4, 5e-4], top_k=3, base_seed=123)
    search = RandOptSearch(model=model, score_fn=score_fn, config=config)
    all_candidates, top_candidates = search.run()

    assert len(all_candidates) == 8
    assert len(top_candidates) == 3
    assert all_candidates == sorted(all_candidates, key=lambda candidate: candidate.score, reverse=True)
    assert top_candidates == all_candidates[:3]

    # Model must be restored after search.
    assert torch.allclose(model.weight, initial_weight)


def test_randopt_parameter_filter():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    config = RandOptConfig(population_size=4, sigma_values=[1e-3], top_k=1, base_seed=0)
    search = RandOptSearch(
        model=model,
        score_fn=lambda _: 0.0,
        config=config,
        parameter_filter=lambda name, _: "0.weight" in name,
    )
    assert len(search._params) == 1
    assert search._params[0][0] == "0.weight"


def test_majority_vote():
    answers = [
        ["a", "x", ""],
        ["a", "y", ""],
        ["b", "x", ""],
    ]
    assert majority_vote(answers) == ["a", "x", ""]
    assert majority_vote(answers, k=2) == ["a", "x", ""]


def test_majority_vote_rejects_mismatched_lengths():
    answers = [["a", "b"], ["a"]]
    try:
        majority_vote(answers)
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("majority_vote should reject ragged input.")
