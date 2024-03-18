# ORPO Trainer

[Odds Ratio Preference Optimization](https://arxiv.org/abs/2403.07691) (ORPO) by Jiwoo Hong, Noah Lee, and James Thorne studies the crucial role of SFT within the context of preference alignment. Using preference data the method 
posits that a minor penalty for the disfavored generation style is sufficient for preference-aligned SFT.

Thus ORPO is a reference model-free  preference optimization algorithm  eliminating the necessity for an additional preference alignment phase. 

The official code can be found [xfactlab/orpo](https://github.com/xfactlab/orpo).

## Expected dataset format

The ORPO trainer expects a format identical to the DPO trainer, which should include three entries. These entries should be named as follows:

- `prompt`
- `chosen`
- `rejected`

for example:

```py
orpo_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Java",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "C++",
    ],
}
```
where the `prompt` contains the context inputs, `chosen` contains the corresponding chosen responses and `rejected` contains the corresponding negative (rejected) responses. As can be seen a prompt can have multiple responses and this is reflected in the entries being repeated in the dictionary's value arrays.

## Expected model format
The ORPO trainer expects a model of `AutoModelForCausalLM`, compared to PPO that expects `AutoModelForCausalLMWithValueHead` for the value function.

## Using the `ORPOTrainer`
For a detailed example have a look at the `examples/scripts/orpo.py` script. At a high level we need to initialize the `ORPOTrainer` with a `model` we wish to train. **Note that ORPOTrainer eliminates the need to use the reference model, simplifying the optimization process.** The `beta` refers to the hyperparameter `lambda` in the paper or `alpha` in the code and refers to the weighting of the relative ratio loss in the full SFT loss.

```py
orpo_config = ORPOConfig(
    beta=0.1, # the lambda/alpha hyperparameter in the paper/code
)

orpo_trainer = ORPOTrainer(
    model,
    args=orpo_config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)
```
After this one can then call:

```py
orpo_trainer.train()
```

## Logging

While training and evaluating we record the following reward metrics:

TODO

## ORPOTrainer

[[autodoc]] ORPOTrainer


## ORPOConfig

[[autodoc]] ORPOConfig
