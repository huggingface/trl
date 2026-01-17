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

import logging

import torch
from transformers import PreTrainedModel, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import CallbackHandler

from ...trainer.callbacks import BEMACallback as _BEMACallback


# Logger for module-level logging
logger = logging.getLogger(__name__)


class CallbackHandlerWithRefModel(CallbackHandler):
    """
    A [`~transformers.CallbackHandler`] that supports passing a reference model to callbacks.
    """

    def __init__(self, callbacks, model, ref_model, processing_class, optimizer, lr_scheduler):
        super().__init__(callbacks, model, processing_class, optimizer, lr_scheduler)
        self.ref_model = ref_model

    # Copied from CallbackHandler.call_event with the addition of `ref_model` to the callback call.
    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                ref_model=self.ref_model,  # <- Added ref_model to the callback call
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class BEMACallback(_BEMACallback):
    # docstyle-ignore
    r"""
    A [`~transformers.TrainerCallback`] that implements [BEMA](https://huggingface.co/papers/2508.00180)
    (Bias-Corrected Exponential Moving Average) by [Adam Block](https://huggingface.co/abblock) and [Cyril
    Zhang](https://huggingface.co/cyrilzhang). Code from https://github.com/abblock/bema under MIT license.

    BEMA computes model weights that scale like:

    $$
    \theta_t' = \alpha_t \cdot (\theta_t - \theta_0) + \text{EMA}_t
    $$

    where  \\( \theta_t \\) is the current model weights,  \\( \theta_0 \\) is a snapshot of the model weights at the
    first `update_after` step,  \\( \text{EMA}_t  \\) is the exponential moving average of the model weights, and
     \\( \alpha_t \\) is a scaling factor that decays with the number of steps  \\( t \\) as

    $$
    \alpha_t = (\rho + \gamma \cdot t)^{-\eta}.
    $$

    The EMA is computed as:

    $$
    \text{EMA}_t = (1 - \beta_t) \cdot \text{EMA}_{t-1} + \beta_t \cdot \theta_t
    $$

    where  \\( \beta_t \\) is a decay factor that decays with the number of steps  \\( t \\) as

    $$
    \beta_t = (\rho + \gamma \cdot t)^{-\kappa}.
    $$

    Args:
        update_freq (`int`, *optional*, defaults to `400`):
            Update the BEMA weights every X steps. Denoted this as  \\( \phi \\) in the paper.
        ema_power (`float`, *optional*, defaults to `0.5`):
            Power for the EMA decay factor. Denoted  \\( \kappa \\) in the paper. To disable EMA, set this to `0.0`.
        bias_power (`float`, *optional*, defaults to `0.2`):
            Power for the BEMA scaling factor. Denoted  \\( \eta \\) in the paper. To disable BEMA, set this to `0.0`.
        lag (`int`, *optional*, defaults to `10`):
            Initial offset in the weight decay schedule that controls early-stage smoothness by acting as a virtual
            starting age for the updates. Denoted as  \\( \rho \\) in the paper.
        update_after (`int`, *optional*, defaults to `0`):
            Burn-in time before starting to update the BEMA weights. Denoted  \\( \tau \\) in the paper.
        multiplier (`float`, *optional*, defaults to `1.0`):
            Initial value for the EMA decay factor. Denoted as  \\( \gamma \\) in the paper.
        min_ema_multiplier (`float`, *optional*, defaults to `0.0`):
            Minimum value for the EMA decay factor.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device to use for the BEMA buffers, e.g. `"cpu"` or `"cuda"`. Note that in most cases, this device SHOULD
            BE DIFFERENT from the device used for training in order to avoid OOM.
        update_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to update the reference model with BEMA weights. This creates a lagged, smoothed version of the
            main model as the reference model.
        ref_model_update_freq (`int`, *optional*, defaults to `400`):
            Update the reference model with BEMA weights every this many steps.
        ref_model_update_after (`int`, *optional*, defaults to `0`):
            Number of steps to wait before starting to update the reference model.

    Example:

    ```python
    from trl import BEMACallback

    trainer = Trainer(..., callbacks=[BEMACallback()])
    ```
    """

    def __init__(
        self,
        update_freq: int = 400,
        ema_power: float = 0.5,
        bias_power: float = 0.2,
        lag: int = 10,
        update_after: int = 0,
        multiplier: float = 1.0,
        min_ema_multiplier: float = 0.0,
        device: str = "cpu",
        update_ref_model: bool = False,
        ref_model_update_freq: int = 400,
        ref_model_update_after: int = 0,
    ):
        super().__init__(
            update_freq,
            ema_power,
            bias_power,
            lag,
            update_after,
            multiplier,
            min_ema_multiplier,
            device,
        )
        # Reference model update parameters
        self.update_ref_model = update_ref_model
        self.ref_model_update_freq = ref_model_update_freq
        self.ref_model_update_after = ref_model_update_after

    @torch.no_grad()
    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs
    ):
        super().on_step_end(args, state, control, model, **kwargs)

        step = state.global_step
        # Update reference model if enabled
        if (
            self.update_ref_model
            and step >= self.ref_model_update_after
            and (step - self.ref_model_update_after) % self.ref_model_update_freq == 0
        ):
            if "ref_model" not in kwargs:
                raise ValueError("'ref_model' not found in kwargs.")

            ref_model = kwargs["ref_model"]

            # Get the current BEMA state dict
            bema_state_dict = self.running_model.state_dict()

            # Handle the case where ref_model is None (PEFT case)
            if ref_model is None:
                # In PEFT case, ref_model is None and we need to update the base model of the main model
                main_model = self._unwrap_model(model)
                if hasattr(main_model, "get_base_model"):
                    # This is a PEFT model, update the base model
                    base_model = main_model.get_base_model()
                    self._update_model_with_bema_weights(base_model, bema_state_dict, is_peft_base=True)
                else:
                    # Regular model, update directly
                    self._update_model_with_bema_weights(main_model, bema_state_dict, is_peft_base=False)
            else:
                # ref_model is provided, unwrap it and update
                ref_model = self._unwrap_model(ref_model)
                if hasattr(ref_model, "get_base_model"):
                    # This is a PEFT model, update the base model
                    base_model = ref_model.get_base_model()
                    self._update_model_with_bema_weights(base_model, bema_state_dict, is_peft_base=True)
                else:
                    # Regular model, update directly
                    self._update_model_with_bema_weights(ref_model, bema_state_dict, is_peft_base=False)

            logger.info("BEMACallback: Updated reference model with BEMA weights")

    def _update_model_with_bema_weights(self, model, bema_state_dict, is_peft_base=False):
        """Helper method to update a model with BEMA weights, handling PEFT and distributed scenarios."""
        if is_peft_base:
            # For PEFT base models, filter out adapter parameters
            filtered_state_dict = {}
            for key, value in bema_state_dict.items():
                # Skip adapter parameters
                if not key.startswith("lora_") and not key.startswith("adapter_"):
                    # Remove 'base_model.' prefix if it exists
                    if key.startswith("base_model."):
                        base_key = key[len("base_model.") :]
                    else:
                        base_key = key
                    filtered_state_dict[base_key] = value

            # Update the base model
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            # Regular model, update directly
            model.load_state_dict(bema_state_dict, strict=False)
