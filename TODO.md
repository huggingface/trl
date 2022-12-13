# TODOs

- [x] `accelerate` support for the trainer
- [ ] custom architectures support (BLOOM, OPT)
- [ ] refactor code
- [ ] remove notebooks
- [ ] confirm if DP works

## Roadmap

How do I imagine the API?
```
from transformers import BloomForCausalLM, BloomTokenizer
from trl import AutoRegressiveLMWithValueHead

base_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

model = AutoRegressiveLMWithValueHead.from_base_model(base_model)
ref_model = AutoRegressiveLMWithValueHead.from_base_model(base_model)


```

## Code organization

- `examples/`: add an example for distributed training
- `trl/accelerate_ppo.py`: `accelerate` trainer
- `trl/base.py`: add base value head.

## How to run it with `accelerate`

```bash
accelerate config
```

Select "no" for every option and run
```bash
accelerate launch main_accelerate.py
```

