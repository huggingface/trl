# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from huggingface_hub import HfApi

from trl.import_utils import is_mergekit_available


if is_mergekit_available():
    from mergekit.config import MergeConfiguration
    from mergekit.merge import MergeOptions, run_merge


def upload_model_to_hf(folder_path: str, repo_id: str):
    api = HfApi()
    # Create the repository if it doesn't exist
    repo = api.create_repo(repo_id, repo_type="model")

    # Upload the folder to the specified repository
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
    )


class MergeConfig:
    r"""
    Configuration class for merging two models using `mergekit`.

    This class provides a structured way to configure and generate merge configurations for various merge methods,
    such as `linear`, `ties`, `dare_ties`, and `slerp`.

    Args:
        method (`str`, *optional*, defaults to `"linear"`):
            Merge method to use. Supported methods include:

            - `"linear"`: Linearly combines two models with specified weights.
            - `"ties"`: Combines two models using the TIES method with density parameters.
            - `"dare_ties"`: A variant of TIES for domain adaptation.
            - `"slerp"`: Combines models using spherical linear interpolation.

    Note:

        For more details about the merge methods and how they are implemented, see the
        [MergeKit GitHub repository](https://github.com/arcee-ai/mergekit?tab=readme-ov-file#merge-methods).

    Attributes:
        method (`str`): The merge method to use.
        policy_model_path (`str` or `None`): Path to the policy model.
        target_model_path (`str` or `None`): Path to the target model.
        policy_model_weight (`float`): Weight for the policy model (for `linear` and `ties` methods).
        target_model_weight (`float`): Weight for the target model (for `linear` and `ties` methods).
        policy_model_density (`list[float]`): Density parameters for the policy model (for `ties` and `dare_ties`).
        target_model_density (`list[float]`): Density parameters for the target model (for `ties` and `dare_ties`).
        normalize (`float` or `None`): Normalization factor for the TIES method.
        t_values (`float` or `None`): Interpolation factor for the SLERP method.
        dtype (`str`): Data type to use for merging, e.g., `"float16"`.
    """

    def __init__(self, method: str = "linear"):
        if not is_mergekit_available():
            raise ImportError(
                "MergeConfig requires the `mergekit` extra. To install, run `pip install trl[mergekit]`."
            )
        self.method = method
        self.policy_model_path = None
        self.target_model_path = None

        # Initialize relevant parameters based on the method
        if method == "linear":
            self.policy_model_weight = 0.5
            self.target_model_weight = 0.5
            self.dtype = "float16"
        elif method == "ties":
            self.policy_model_weight = 1.0
            self.policy_model_density = [1.0, 0.7, 0.1]
            self.target_model_weight = 1.0
            self.target_model_density = [1.0]
            self.normalize = 1.0
            self.dtype = "float16"
        elif method == "dare_ties":
            self.policy_model_weight = 1.0
            self.policy_model_density = [1.0, 0.7, 0.1]
            self.target_model_weight = 1.0
            self.target_model_density = [1.0]
            self.normalize = 1.0
            self.dtype = "float16"
        elif method == "slerp":
            self.t_values = 0.5
            self.dtype = "float16"
        else:
            raise ValueError(f"Unsupported merge method: {method}")

    def create_merge_config_linear(self) -> "MergeConfiguration":
        """
        Creates a merge configuration for a linear merge of two models with specified weights.
        """
        # Create the merge configuration dictionary
        merge_config_dict = {
            "dtype": self.dtype,
            "merge_method": "linear",
            "models": [
                {"model": self.policy_model_path, "parameters": {"weight": self.policy_model_weight}},
                {"model": self.target_model_path, "parameters": {"weight": self.target_model_weight}},
            ],
        }

        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)

        return merge_config

    def create_merge_config_ties(self) -> "MergeConfiguration":
        """
        Creates a merge configuration for a TIES merge of two models, with specified weights and densities.
        """
        # Create the TIES merge configuration dictionary
        merge_config_dict = {
            "merge_method": "ties",
            "slices": None,  # Optional slices if needed
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": {"density": self.target_model_density, "weight": self.target_model_weight},
                },
                {
                    "model": {
                        "model": {"path": self.policy_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": {"density": self.policy_model_density, "weight": self.policy_model_weight},
                },
            ],
            "parameters": {"normalize": self.normalize},
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None,
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None,
        }

        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)

        return merge_config

    def create_merge_config_dare_ties(self) -> "MergeConfiguration":
        """
        Creates a merge configuration for a DARE TIES merge of two models, with specified weights and densities.
        """
        # Create the DARE TIES merge configuration dictionary
        merge_config_dict = {
            "merge_method": "dare_ties",
            "slices": None,  # Optional slices if needed
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": {"density": self.target_model_density, "weight": self.target_model_weight},
                },
                {
                    "model": {
                        "model": {"path": self.policy_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": {"density": self.policy_model_density, "weight": self.policy_model_weight},
                },
            ],
            "parameters": {"normalize": self.normalize},
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None,
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None,
        }

        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)

        return merge_config

    def create_merge_config_slerp(self) -> "MergeConfiguration":
        """
        Creates a merge configuration for a SLERP merge of a model with a base model.
        """

        # Create the SLERP merge configuration dictionary
        merge_config_dict = {
            "merge_method": "slerp",
            "slices": None,  # Optional slices if needed
            "models": [
                {
                    "model": {
                        "model": {"path": self.target_model_path, "revision": None},
                        "lora": None,
                        "override_architecture": None,
                    },
                    "parameters": None,  # No specific parameters for SLERP model
                }
            ],
            "parameters": {
                "t": self.t_values  # Set the t values for SLERP
            },
            "base_model": {
                "model": {"path": self.policy_model_path, "revision": None},
                "lora": None,
                "override_architecture": None,
            },
            "dtype": self.dtype,
            "tokenizer_source": None,
            "tokenizer": None,
            "chat_template": None,
            "out_dtype": None,
        }

        # Create the MergeConfiguration from the dictionary
        merge_config = MergeConfiguration.model_validate(merge_config_dict)

        return merge_config

    def create(self) -> "MergeConfiguration":
        if self.method == "linear":
            return self.create_merge_config_linear()
        elif self.method == "ties":
            return self.create_merge_config_ties()
        elif self.method == "dare_ties":
            return self.create_merge_config_dare_ties()
        elif self.method == "slerp":
            return self.create_merge_config_slerp()


def merge_models(config: MergeConfig, out_path: str):
    """
    Merge two models using mergekit

    Args:
        config (`MergeConfig`): The merge configuration.
        out_path (`str`): The output path for the merged model.
    """
    if not is_mergekit_available():
        raise ImportError("merge_models requires the `mergekit` extra. To install, run `pip install trl[mergekit]`.")
    run_merge(
        config,
        out_path=out_path,
        options=MergeOptions(
            cuda=torch.cuda.is_available(),
            copy_tokenizer=True,
            lazy_unpickle=False,
            low_cpu_memory=False,
        ),
    )
