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

import random
from .utils import pad
import numpy as np

def repad(list_of_tensor_dicts, padding_value):   
    p_ids, p_attn_masks = remove_and_pad(
        [tensor_dict['prompt_ids'] for tensor_dict in list_of_tensor_dicts],
        [tensor_dict['prompt_mask'] for tensor_dict in list_of_tensor_dicts],
        pad_token_id=padding_value,
        padding_side='left',
    )
    c_ids, c_attn_masks = remove_and_pad(
        [tensor_dict['completion_ids'] for tensor_dict in list_of_tensor_dicts],
        [tensor_dict['completion_mask'] for tensor_dict in list_of_tensor_dicts],
        pad_token_id=padding_value
    )
    old_logps, _ = remove_and_pad(
        [tensor_dict['old_per_token_logps'] for tensor_dict in list_of_tensor_dicts],
        [tensor_dict['completion_mask'] for tensor_dict in list_of_tensor_dicts],
        pad_token_id=-10000.0, # ignored so can be anything
    )
    ref_logps, _ = remove_and_pad(
        [tensor_dict['ref_per_token_logps'] for tensor_dict in list_of_tensor_dicts],
        [tensor_dict['completion_mask'] for tensor_dict in list_of_tensor_dicts],
        pad_token_id=-10000.0, # ignored so can be anything
    )
    
    for i, (p_id, p_mask, c_id, c_mask, o_logp, r_logp) in enumerate(zip(p_ids, p_attn_masks, c_ids, c_attn_masks, old_logps, ref_logps)):        
        list_of_tensor_dicts[i]['prompt_ids'] = p_id
        list_of_tensor_dicts[i]['prompt_mask'] = p_mask
        list_of_tensor_dicts[i]['completion_ids'] = c_id
        list_of_tensor_dicts[i]['completion_mask'] = c_mask
        list_of_tensor_dicts[i]['old_per_token_logps'] = o_logp
        list_of_tensor_dicts[i]['ref_per_token_logps'] = r_logp
        
    return list_of_tensor_dicts
    
def remove_and_pad(list_of_ids, list_of_masks, pad_token_id=0, padding_side='right'):
    """
    Remove padding from list_of_ids and list_of_masks, and then pad them to the same length.
    """
    num_samples = len(list_of_ids)
    if list_of_ids[0] is None:
        # we are not using old_per_token_logps / ref_per_token_logps
        return [None]*num_samples, [None]*num_samples
    # Remove padding
    list_of_ids = [ids[mask == 1] for ids, mask in zip(list_of_ids, list_of_masks)]
    list_of_masks = [mask[mask == 1] for mask in list_of_masks]
    
    ids = pad(list_of_ids, padding_value=pad_token_id, padding_side=padding_side)
    masks = pad(list_of_masks, padding_value=0, padding_side=padding_side)
    
    return ids, masks

def remove_padding(input_ids, attn_mask):
    """
    Remove padding from input_ids and attn_mask.
    """
    if attn_mask is not None:
        input_ids = input_ids[attn_mask == 1]
        attn_mask = attn_mask[attn_mask == 1]
    return input_ids, attn_mask
    


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.sample_indices = []

    def add(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)

        # Clear index queue when buffer changes
        self.sample_indices.clear()

    def _init_sampling_queue(self):
        self.sample_indices = list(range(len(self.buffer)))
        random.shuffle(self.sample_indices)

    def sample(self, batch_size):
        if not self.sample_indices:
            self._init_sampling_queue()

        batch = []
        while len(batch) < batch_size and self.sample_indices:
            idx = self.sample_indices.pop(0)
            batch.append(self.buffer[idx])

        if len(batch) != batch_size:
            raise ValueError("Not enough samples in the buffer to fill the batch.")

        return batch

    def __len__(self):
        return len(self.buffer)
    
    
class SSRReplayBuffer(ReplayBuffer):
    # implementation of the SSR replay buffer from https://arxiv.org/pdf/2504.08837
    def __init__(self, capacity, alpha=1.0):
        super().__init__(capacity)
        self.alpha = alpha
        self.advantages = []

    def add(self, experience):
        EPS = 0.0001 # ensures we get non-zero advs when the buffer contains all 0 advantages
        advantage = experience["advantages"].item()
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.advantages.append(abs(advantage) + EPS)  # Store absolute advantage
        else:
            # Replace the oldest entry if the buffer is full
            self.buffer.pop(0)
            self.advantages.pop(0)
            self.buffer.append(experience)
            self.advantages.append(abs(advantage))

    def sample(self, batch_size):
        if not self.buffer:
            raise ValueError("Buffer is empty. Cannot sample from an empty buffer.")

        # Convert advantages to priorities
        scaled_priorities = np.power(self.advantages, self.alpha)
        total_priority = np.sum(scaled_priorities)
        probabilities = scaled_priorities / total_priority

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[i] for i in indices]