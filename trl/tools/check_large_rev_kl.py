
import os
import torch
from transformers import AutoTokenizer

def get_rev_kl(log_p: torch.Tensor, log_q: torch.Tensor, mask: torch.Tensor):
    log_ratio = (log_p - log_q) * mask
    kl = log_ratio.float().exp() - 1 - log_ratio
    return kl

input_dir = "results/train/minillm/openr1-math-220k/qwen2.5-1.5B/lr5e-6_hf_l256_bs256_test2/large_reverse_kl_logs/rank_4/gs2_s11_rkl_5.87/"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

inputs = torch.load(os.path.join(input_dir, "inputs.pt"), map_location="cpu")
input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
labels = input_ids.clone()
labels[attention_mask == 0] = -100
prompt_lengths = inputs["prompt_ids"].shape[1]
shifted_labels = labels[:, prompt_lengths:]
mask = shifted_labels != -100

generated_lengths = torch.sum(mask, dim=1)

print(generated_lengths)

for i, inp in enumerate(inputs):
    print(f"Input {i}: {inp}")
    
print(inputs["prompt_ids"].shape)
print(tokenizer.batch_decode(inputs["prompt_ids"], skip_special_tokens=True))
print(inputs["completion_ids"].shape)
print(tokenizer.batch_decode(inputs["completion_ids"], skip_special_tokens=True))

teacher_log_probs = torch.load(os.path.join(input_dir, "teacher_log_probs.pt"), map_location="cpu")
student_log_probs = torch.load(os.path.join(input_dir, "student_log_probs.pt"), map_location="cpu")

teacher_log_probs_on_labels = torch.gather(
    teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
).squeeze(-1)
student_log_probs_on_labels = inputs["old_per_token_logps"]

per_token_reverse_kl = get_rev_kl(
    log_p=teacher_log_probs_on_labels, log_q=student_log_probs_on_labels, mask=mask
)
reverse_kl = per_token_reverse_kl.sum(dim=1) / mask.sum(dim=1)
print(per_token_reverse_kl)
print(reverse_kl)
# print(list(map(lambda x: round(x, 4), per_token_reverse_kl[2].tolist())))
# print(list(map(lambda x: round(x, 4), student_log_probs_on_labels[2].tolist())))
for i, x in enumerate(per_token_reverse_kl[6]):
    if x > 1000:
        print(i)
        

print(per_token_reverse_kl[6][34])
print(teacher_log_probs_on_labels[6][34])
print(student_log_probs_on_labels[6][34])
print(shifted_labels[6][34-10:34+10])
print(tokenizer.convert_ids_to_tokens(shifted_labels[6][34-10:34+10].tolist()))
print(tokenizer.convert_ids_to_tokens([shifted_labels[6][34]]))
print(inputs["old_per_token_logps"].shape)
print(inputs["sampling_per_token_logps"].shape)

print(inputs["old_per_token_logps"][6][34])
print(inputs["sampling_per_token_logps"][6][34])

print(inputs["old_per_token_logps"].dtype)
print(inputs["sampling_per_token_logps"].dtype)

print(inputs["old_per_token_logps"][6] / inputs["sampling_per_token_logps"][6])