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

from huggingface_hub import whoami


model_name = "unsloth/Llama-3.2-3B"
tokenizer_name = "unsloth/Llama-3.2-3B"
dataset_name = "WillHeld/top_v2"

output_root_dir = "./checkpoints/"
hub_model_id = f"{whoami()['name']}/layerskip-{model_name.split('/')[1]}-{dataset_name.split('/')[1]}"
output_dir = f"{output_root_dir}/{hub_model_id}"

per_device_train_batch_size = 8
gradient_accumulation_steps = 1
learning_rate = 2e-5
