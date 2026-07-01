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

from ...trainer.dpo_trainer import DPOTrainer as _DPOTrainer
from .callback import CallbackHandlerWithRefModel


class DPOTrainer(_DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace with a new one that calls the events with the reference model
        self.callback_handler = CallbackHandlerWithRefModel(
            self.callback_handler.callbacks,
            self.model,
            self.ref_model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )
