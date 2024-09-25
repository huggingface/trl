---
{{ card_data }}
---

# Model Card for {{ model_name }}

This model is a fine-tuned version of [{{ base_model }}](https://huggingface.co/{{ base_model }}){% if dataset_name %} on the [{{ dataset_name }}](https://huggingface.co/datasets/{{ dataset_name }}) dataset{% endif %}.
It has been trained using [TRL](https://github.com/huggingface/trl).

## Training procedure

This model was trained with {{ trainer_name }}{% if paper_id %}, a method introduced in [{{ paper_title }}](https://huggingface.co/papers/{{ paper_id }}){% endif %}.

### Framework versions

- TRL: {{ trl_version }}
- Transformers: {{ transformers_version }}
- Pytorch: {{ pytorch_version }}
- Datasets: {{ datasets_version }}
- Tokenizers: {{ tokenizers_version }}
