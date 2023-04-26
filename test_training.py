import torch
from peft import get_peft_model, LoraConfig, TaskType

from transformers import AutoModelForSequenceClassification

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model_id = "gpt2"

model = AutoModelForSequenceClassification.from_pretrained(model_id)

dummy_input = torch.LongTensor([[1, 2, 3, 4]])

peft_model = get_peft_model(model, peft_config)

output_1 = peft_model(dummy_input)[0]
output_2 = peft_model(dummy_input)[0]
loss = (output_1 + output_2).sum()

loss.backward()

for n, param in peft_model.named_parameters():
    if 'lora' in n:
        assert param.grad is not None