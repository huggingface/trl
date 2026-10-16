import copy

import pytest
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from trl import DistillationConfig, DistillationTrainer, RLOOConfig, RLOOTrainer


@pytest.mark.parametrize("pan_and_scan", [False, True])
@pytest.mark.parametrize("trainer_kind", ["rloo", "distillation"])
def test_train_gemma_pan_and_scan(tmp_path, pan_and_scan, trainer_kind):
    model_id = "trl-internal-testing/tiny-Gemma3ForConditionalGeneration"
    processor = AutoProcessor.from_pretrained(
        model_id,
        image_seq_length=4,
        do_pan_and_scan=pan_and_scan,
        pan_and_scan_min_crop_size=32,
        pan_and_scan_max_num_crops=4,
        pan_and_scan_min_ratio_to_activate=1.2,
    )
    processor.image_processor.size = {"height": 28, "width": 28}
    config = AutoConfig.from_pretrained(model_id)
    config.vision_config.image_size = 28
    config.mm_tokens_per_image = 4
    config.text_config.head_dim = 8
    config.text_config.query_pre_attn_scalar = 8
    model = AutoModelForImageTextToText.from_config(config, attn_implementation="eager")
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "Describe the image."}]] * 4,
            "image": [
                Image.new("RGB", size, color)
                for size, color in [
                    ((128, 32), "red"),
                    ((32, 32), "blue"),
                    ((32, 128), "green"),
                    ((32, 32), "white"),
                ]
            ],
        }
    )
    config_kwargs = dict(
        output_dir=str(tmp_path),
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_completion_length=2,
        max_steps=2,
        learning_rate=0.01,
        report_to="none",
        save_strategy="no",
        seed=42,
    )
    if trainer_kind == "rloo":
        initial_model_path = tmp_path / "initial_model"
        model.save_pretrained(initial_model_path)
        model.config._name_or_path = str(initial_model_path)

        def reward_func(completions, **kwargs):
            return [float(i % 2) for i in range(len(completions))]

        trainer = RLOOTrainer(
            model=model,
            processing_class=processor,
            args=RLOOConfig(**config_kwargs, generation_batch_size=4, num_generations=2),
            train_dataset=dataset,
            reward_funcs=reward_func,
        )
    else:
        trainer = DistillationTrainer(
            model=model,
            teacher_model=copy.deepcopy(model),
            processing_class=processor,
            args=DistillationConfig(**config_kwargs),
            train_dataset=dataset,
        )
    before = model.model.multi_modal_projector.mm_input_projection_weight.detach().clone()
    result = trainer.train()
    assert trainer.state.global_step == 2
    assert torch.isfinite(torch.tensor(result.training_loss))
    if trainer_kind == "rloo":
        assert not torch.equal(before, model.model.multi_modal_projector.mm_input_projection_weight.detach())
