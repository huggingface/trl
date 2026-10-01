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

import pytest
from datasets import Dataset, Features, Image, IterableDataset, Value
from transformers.utils import is_vision_available

from trl import DistillationConfig, DistillationTrainer, GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer
from trl.data_utils import _has_vision_data

from .testing_utils import TrlTestCase, get_vision_parameter_names, require_vision


if is_vision_available():
    from PIL import Image as PILImage


@pytest.mark.parametrize(
    "features,expected",
    [
        (Features({"prompt": Value("string")}), False),
        (Features({"prompt": [{"role": Value("string"), "content": Value("string")}]}), False),
        (Features({"prompt": [{"content": [{"type": Value("string"), "text": Value("string")}]}]}), False),
        (Features({"prompt": Value("string"), "image": Image()}), True),
        (Features({"prompt": Value("string"), "images": [Image()]}), True),
        (Features({"prompt": [{"content": [{"type": Value("string"), "image": Image()}]}]}), True),
        (Features({"prompt": [{"content": [{"type": Value("string"), "image": Value("string")}]}]}), True),
        (None, True),
    ],
)
def test_vision_schema_detection_does_not_read_stream(features, expected):
    def generate():
        raise AssertionError("Vision schema detection must not read streaming data")
        yield

    dataset = IterableDataset.from_generator(generate, features=features)
    assert _has_vision_data(dataset) is expected


@require_vision
class TestEmbeddedImagesKeepVisionTrainable(TrlTestCase):
    @pytest.mark.parametrize("trainer_name", ["grpo", "rloo", "distillation"])
    @pytest.mark.parametrize("dataset_kind", ["image_only", "mixed", "typed_stream", "untyped_stream"])
    def test_embedded_images_keep_vision_parameters_trainable(self, trainer_name, dataset_kind):
        features = Features(
            {
                "prompt": [
                    {
                        "role": Value("string"),
                        "content": [{"type": Value("string"), "image": Image(), "text": Value("string")}],
                    }
                ]
            }
        )
        text_row = {"prompt": [{"role": "user", "content": [{"type": "text", "text": "Hello", "image": None}]}]}
        image_row = {
            "prompt": [
                {"role": "user", "content": [{"type": "image", "image": PILImage.new("RGB", (32, 32)), "text": None}]}
            ]
        }
        # A text-only first row must not hide images later in a mixed dataset.
        rows = [image_row] * 4 if dataset_kind == "image_only" else [text_row, image_row] * 2
        dataset = Dataset.from_list(rows, features=features)
        if dataset_kind == "typed_stream":
            dataset = dataset.to_iterable_dataset()
        elif dataset_kind == "untyped_stream":
            dataset = IterableDataset.from_generator(lambda: iter(rows))

        model_id = "trl-internal-testing/tiny-LlavaForConditionalGeneration"
        config_kwargs = {
            "output_dir": self.tmp_dir,
            "report_to": "none",
            "bf16": False,
            "gradient_checkpointing": False,
            "per_device_train_batch_size": 2,
            "max_steps": 1,
        }
        if trainer_name == "distillation":
            trainer = DistillationTrainer(
                model=model_id,
                teacher_model=model_id,
                args=DistillationConfig(**config_kwargs),
                train_dataset=dataset,
            )
        else:
            trainer_cls, config_cls = (
                (GRPOTrainer, GRPOConfig) if trainer_name == "grpo" else (RLOOTrainer, RLOOConfig)
            )
            trainer = trainer_cls(
                model=model_id,
                reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
                args=config_cls(**config_kwargs, num_generations=2),
                train_dataset=dataset,
            )

        vision_parameter_names = get_vision_parameter_names(trainer.model)
        assert vision_parameter_names
        assert all(trainer.model.get_parameter(name).requires_grad for name in vision_parameter_names)
        # Confirm these embedded images actually reach the multimodal processor.
        assert trainer._tokenize_prompts([image_row["prompt"]])[1] is not None
