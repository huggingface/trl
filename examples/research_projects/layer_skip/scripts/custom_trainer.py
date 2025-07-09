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

from trl import SFTTrainer


class LayerSkipSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_exit_layer = 0  # initialize with 0
        self.always_last_layer = True
        self.early_exit_loss_scale = 1.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.early_exit_layer = (
            self.early_exit_layer % (model.config.num_hidden_layers - 1)
        ) + 1  # rotates between [1, num_hidden_layers-1]
        bs, seqlen = inputs.input_ids.shape

        labels = inputs.pop("labels")
        outputs = model(**inputs, output_hidden_states=True)

        hidden_state = outputs["hidden_states"][self.early_exit_layer].to(model.dtype)
        if self.early_exit_layer != model.config.num_hidden_layers:
            hidden_state = model.model.norm(hidden_state)
        logits = model.lm_head(hidden_state)
        loss_early = model.loss_function(logits=logits, labels=labels, vocab_size=model.vocab_size)

        if self.always_last_layer:
            loss_last = model.loss_function(logits=outputs["logits"], labels=labels, vocab_size=model.vocab_size)
            loss = self.early_exit_loss_scale * loss_early.to(loss_last.device) + 1.0 * loss_last
            # normalize loss scales
            loss = loss / (1.0 + self.early_exit_loss_scale)
        else:
            loss = loss_early

        return loss
