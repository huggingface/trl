---
{{ card_data }}
---

# Model Card for {{ model_name }}

This model is a fine-tuned version of [{{ base_model }}](https://huggingface.co/{{ base_model }}){% if dataset_name %} on the [{{ dataset_name }}](https://huggingface.co/datasets/{{ dataset_name }}) dataset{% endif %}.
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="{{ hub_model_id }}", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

{% if wandb_url %}[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>]({{ wandb_url }}){% endif %} 
{% if comet_url %}[<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/master/logo/comet_badge.png" alt="Visualize in Comet" width="135" height="20"/>]({{ comet_url }}){% endif %}

This model was trained with {{ trainer_name }}{% if paper_id %}, a method introduced in [{{ paper_title }}](https://huggingface.co/papers/{{ paper_id }}){% endif %}.

### Framework versions

- TRL: {{ trl_version }}
- Transformers: {{ transformers_version }}
- Pytorch: {{ pytorch_version }}
- Datasets: {{ datasets_version }}
- Tokenizers: {{ tokenizers_version }}

## Citations

{% if trainer_citation %}Cite {{ trainer_name }} as:

```bibtex
{{ trainer_citation }}
```{% endif %}

Cite TRL as:
    
```bibtex
{% raw %}@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}{% endraw %}
```
