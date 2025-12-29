# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field

from trl.trainer.grpo_config import GRPOConfig


@dataclass
class HICRAConfig(GRPOConfig):
    r"""
    Configuration class for the [`HICRATrainer`].

    This class extends [`GRPOConfig`] with HICRA-specific parameters for Hierarchy-Aware Credit Assignment.
    HICRA amplifies learning signals for strategic planning tokens, enabling LLMs to develop hierarchical
    reasoning capabilities more efficiently than standard GRPO.

    For a full list of training arguments, please refer to the [`~transformers.TrainingArguments`] and
    [`GRPOConfig`] documentation.

    Parameters:
        > HICRA-specific parameters

        hicra_alpha (`float`, *optional*, defaults to `0.2`):
            Amplification factor for planning/high-entropy tokens. This controls how much to amplify the
            advantages for tokens identified as strategic planning tokens. The paper "Emergent Hierarchical
            Reasoning in LLMs through Reinforcement Learning" (arXiv:2509.03646) uses α=0.2.
        use_hicra (`bool`, *optional*, defaults to `True`):
            Whether to enable HICRA advantage modification. If set to `False`, the trainer behaves exactly
            like standard GRPO.
        hicra_entropy_topk (`float`, *optional*, defaults to `0.3`):
            Top-k percentile for entropy threshold computation. Tokens with entropy above the (1-k) percentile
            are considered high-entropy tokens and may be amplified. The VeRL implementation uses k=0.3.
        use_planning_tokens (`bool`, *optional*, defaults to `False`):
            Whether to use Strategic Gram-based planning token identification. If `True`, tokens that are part
            of Strategic Grams will be amplified. If `False`, only high-entropy tokens are amplified.

        > Strategic Gram configuration

        strategic_grams_path (`str`, *optional*):
            Path to a JSON file containing pre-computed Strategic Grams. If provided, these Strategic Grams
            will be loaded and used for planning token identification. Mutually exclusive with `strategic_grams`.
        strategic_grams (`list[str]`, *optional*):
            Direct list of Strategic Gram strings to use for planning token identification. If provided, these
            will be used instead of loading from a file. Mutually exclusive with `strategic_grams_path`.
        sg_n_range (`tuple[int, int]`, *optional*, defaults to `(3, 5)`):
            N-gram range for Strategic Gram extraction. When extracting Strategic Grams from a corpus, this
            specifies the minimum and maximum n-gram lengths to consider.

        > Logging configuration

        log_semantic_entropy (`bool`, *optional*, defaults to `True`):
            Whether to log semantic entropy metrics. Semantic entropy measures the diversity of Strategic Grams
            used in generated completions, providing insight into strategic exploration.
        log_planning_token_ratio (`bool`, *optional*, defaults to `True`):
            Whether to log the ratio of planning tokens to total tokens. This metric helps monitor how much of
            the model's output consists of strategic planning vs. execution.
    """

    # HICRA-specific parameters
    hicra_alpha: float = field(
        default=0.2,
        metadata={
            "help": "Amplification factor for planning/high-entropy tokens. Controls how much to amplify "
            "advantages for strategic planning tokens. The paper uses α=0.2."
        },
    )
    use_hicra: bool = field(
        default=True,
        metadata={"help": "Whether to enable HICRA advantage modification. If False, behaves like standard GRPO."},
    )
    hicra_entropy_topk: float = field(
        default=0.3,
        metadata={
            "help": "Top-k percentile for entropy threshold computation. Tokens with entropy above the (1-k) "
            "percentile are considered high-entropy tokens. The VeRL implementation uses k=0.3."
        },
    )
    use_planning_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to use Strategic Gram-based planning token identification. If True, tokens that "
            "are part of Strategic Grams will be amplified. If False, only high-entropy tokens are amplified."
        },
    )

    # Strategic Gram configuration
    strategic_grams_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to a JSON file containing pre-computed Strategic Grams. If provided, these will be "
            "loaded and used for planning token identification. Mutually exclusive with `strategic_grams`."
        },
    )
    strategic_grams: list[str] | None = field(
        default=None,
        metadata={
            "help": "Direct list of Strategic Gram strings to use for planning token identification. If provided, "
            "these will be used instead of loading from a file. Mutually exclusive with `strategic_grams_path`."
        },
    )
    sg_n_range: tuple[int, int] = field(
        default=(3, 5),
        metadata={
            "help": "N-gram range for Strategic Gram extraction. Specifies the minimum and maximum n-gram "
            "lengths to consider when extracting Strategic Grams from a corpus."
        },
    )

    # Logging configuration
    log_semantic_entropy: bool = field(
        default=True,
        metadata={
            "help": "Whether to log semantic entropy metrics. Semantic entropy measures the diversity of "
            "Strategic Grams used in generated completions."
        },
    )
    log_planning_token_ratio: bool = field(
        default=True,
        metadata={
            "help": "Whether to log the ratio of planning tokens to total tokens. This metric helps monitor "
            "how much of the model's output consists of strategic planning vs. execution."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Validate HICRA-specific parameters
        if self.hicra_alpha < 0 or self.hicra_alpha > 1:
            raise ValueError(
                f"hicra_alpha must be in the range [0, 1], but got {self.hicra_alpha}. The paper recommends α=0.2."
            )

        if self.hicra_entropy_topk < 0 or self.hicra_entropy_topk > 1:
            raise ValueError(
                f"hicra_entropy_topk must be in the range [0, 1], but got {self.hicra_entropy_topk}. "
                "The VeRL implementation uses k=0.3."
            )

        if self.strategic_grams_path is not None and self.strategic_grams is not None:
            raise ValueError(
                "strategic_grams_path and strategic_grams are mutually exclusive. Please provide only one of them."
            )

        if len(self.sg_n_range) != 2 or self.sg_n_range[0] > self.sg_n_range[1]:
            raise ValueError(
                f"sg_n_range must be a tuple of (min_n, max_n) where min_n <= max_n, but got {self.sg_n_range}."
            )
