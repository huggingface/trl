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

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# LoRA
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import argparse
import os


# Per-rank Triton/Inductor cache to avoid file-not-found races when 16 ranks share
# the default `/fsx` triton cache. Must run before any torch.compile / triton import.
if "LOCAL_RANK" in os.environ:
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        f"/tmp/triton-rank-{os.environ.get('RANK', '0')}-{os.uname().nodename}",
    )

# Workaround for PyTorch 2.10+ inductor crash with torch.compile + TF32.
# Transformers' enable_tf32() sets torch.backends.fp32_precision (new API), but
# inductor reads torch.backends.cuda.matmul.allow_tf32 (legacy API), and PyTorch
# errors on mixed API usage. Setting the legacy flags first makes both APIs agree.
import torch  # noqa: E402


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(script_args, training_args, model_args, dataset_args):
    from accelerate import logging
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

    from trl import SFTTrainer, get_dataset, get_kbit_device_map, get_peft_config, get_quantization_config

    logger = logging.get_logger(__name__)

    ################
    # Model init kwargs
    ################
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )

    # With EP, transformers shard resolution races 16+ ranks × 16 shards through hub HEAD
    # requests; some get rate-limited, raising spurious OSErrors. HF_HUB_OFFLINE=1 fixes the
    # model load but blocks the kernels package's hub `get_kernel` calls. So: pre-warm the
    # sonicmoe kernel (which uses @functools.cache, only one path), then flip offline.
    # Only enable this dance for EP runs — non-EP runs don't hit the race and need hub access
    # for FA3's `load_and_register_attn_kernel` which has its own get_kernel path the pre-warm
    # doesn't cover.
    import os as _os_kernel_warm

    if training_args.enable_expert_parallel:
        if getattr(training_args, "experts_implementation", None) == "sonicmoe":
            from transformers.integrations.sonicmoe import _load_sonic_kernel

            _load_sonic_kernel()  # @functools.cache → kernel cached in this process
        # FA3 (or any kernel-based attn) is loaded via two separate paths during from_pretrained:
        # load_and_register_attn_kernel (registers in ALL_ATTENTION_FUNCTIONS) and
        # lazy_import_flash_attention (sets module-level _flash_fn etc). Both hit the hub —
        # pre-warm both so the offline flip below doesn't break either.
        if model_args.attn_implementation and "/" in model_args.attn_implementation:
            from transformers.integrations.hub_kernels import load_and_register_attn_kernel
            from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

            load_and_register_attn_kernel(model_args.attn_implementation)
            lazy_import_flash_attention(model_args.attn_implementation)
        # Monkey-patch the already-loaded module constant — env var alone is too late.
        _os_kernel_warm.environ["HF_HUB_OFFLINE"] = "1"
        import huggingface_hub.constants as _hf_const

        _hf_const.HF_HUB_OFFLINE = True

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Create model
    # When enable_expert_parallel is set, pass model as string to SFTTrainer so it handles
    # EP device mesh creation and distributed_config inside from_pretrained.
    if training_args.enable_expert_parallel:
        # Pass model as string so SFTTrainer handles EP device mesh + distributed_config.
        model = model_args.model_name_or_path
        training_args.model_init_kwargs = model_kwargs
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

        if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    if training_args.save_strategy != "no":
        trainer.save_model(training_args.output_dir)
        trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None, prog: str | None = None):
    from trl import DatasetMixtureConfig, ModelConfig, ScriptArguments, SFTConfig, TrlParser

    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types, prog=prog)
    return parser


def _preparse_dict_args() -> None:
    """JSON-decode `--liger_kernel_config` BEFORE HfArgumentParser sees it.

    HfArgumentParser falls back to `type=dict` for `dict[str, bool] | None` fields,
    and `dict("{...}")` raises. Pre-replacing the arg with a Python-evaluated dict
    won't help (argparse still passes a string). The simplest fix: extract the
    JSON string from sys.argv, store the parsed dict in an env var, then drop the
    flag from argv. We re-inject after HfArgumentParser is done.
    """
    import sys

    for flag in ("--liger_kernel_config", "--liger-kernel-config"):
        if flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 < len(sys.argv):
                os.environ["_PREPARSED_LIGER_KERNEL_CONFIG"] = sys.argv[i + 1]
                del sys.argv[i : i + 2]


if __name__ == "__main__":
    _preparse_dict_args()
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    if "_PREPARSED_LIGER_KERNEL_CONFIG" in os.environ:
        import json

        training_args.liger_kernel_config = json.loads(os.environ.pop("_PREPARSED_LIGER_KERNEL_CONFIG"))
    main(script_args, training_args, model_args, dataset_args)
