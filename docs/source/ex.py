import torch

from trl import DPOTrainer
from trl.trainer.utils import selective_log_softmax


class MyDPOTrainer(DPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # In this example: the hinge loss from https://huggingface.co/papers/2309.06657
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        shift_labels, shift_completion_mask = input_ids[:, 1:], inputs["completion_mask"][:, 1:]
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Sum the log-probs over the completion tokens, for the policy and for the reference model
        logps = (selective_log_softmax(logits[:, :-1], shift_labels) * shift_completion_mask).sum(dim=1)
        ref_logps = (selective_log_softmax(ref_logits[:, :-1], shift_labels) * shift_completion_mask).sum(dim=1)

        chosen_logps, rejected_logps = logps.chunk(2, dim=0)  # batch is [chosen, rejected]
        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)
        delta = (chosen_logps - ref_chosen_logps) - (rejected_logps - ref_rejected_logps)
        return torch.relu(1 - self.beta * delta).mean()
