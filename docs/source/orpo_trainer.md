# ORPO Trainer

[Odds Ratio Preference Optimization](https://arxiv.org/abs/2403.07691) (ORPO) by Jiwoo Hong, Noah Lee, and James Thorne studies the crucial role of SFT within the context of preference alignment. Using preference data the method posits that a minor penalty for the disfavored generation together with a strong adaption signal to the chosen response via a simple log odds ratio term appended to the NLL loss is sufficient for preference-aligned SFT.

Thus ORPO is a reference model-free preference optimization algorithm eliminating the necessity for an additional preference alignment phase thus saving compute and memory.

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
where the `prompt` contains the context inputs, `chosen` contains the corresponding chosen responses and `rejected` contains the corresponding negative (rejected) responses. Note that a prompt can have multiple responses and this is reflected in the entries being repeated in the dictionary's value arrays.

## Expected model format
The ORPO trainer expects a model of `AutoModelForCausalLM`, compared to PPO that expects `AutoModelForCausalLMWithValueHead` for the value function.

## Using the `ORPOTrainer`
For a detailed example have a look at the `examples/scripts/orpo.py` script. At a high level we need to initialize the `ORPOTrainer` with a `model` we wish to train. **Note that ORPOTrainer eliminates the need to use the reference model, simplifying the optimization process.** The `beta` refers to the hyperparameter `lambda` in eq. (6) of the paper and refers to the weighting of the relative odd ratio loss in the standard cross-entropy loss used for SFT.

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

### For Mixture of Experts Models: Enabling the auxiliary loss

MOEs are the most efficient if the load is about equally distributed between experts.  
To ensure that we train MOEs similarly during preference-tuning, it is beneficial to add the auxiliary loss from the load balancer to the final loss.  

This option is enabled by setting `output_router_logits=True` in the model config (e.g. MixtralConfig).  
To scale how much the auxiliary loss contributes to the total loss, use the hyperparameter `router_aux_loss_coef=...` (default: 0.001).

## Logging

While training and evaluating we record the following reward metrics:

* `rewards/chosen`: the mean log probabilities of the policy model for the chosen responses scaled by beta
* `rewards/rejected`: the mean log probabilities of the policy model for the rejected responses scaled by beta
* `rewards/accuracies`: mean of how often the chosen rewards are > than the corresponding rejected rewards
* `rewards/margins`: the mean difference between the chosen and corresponding rejected rewards

* `log_odds_chosen`: the mean log odds ratio of the chosen responses over the rejected responses

* `log_odds_ratio`: the mean of the `log(sigmoid(log_odds_chosen))`

* `nll_loss`: the mean negative log likelihood loss from the SFT part of the loss over chosen responses
 
## ORPOTrainer

[[autodoc]] ORPOTrainer


## ORPOConfig

[[autodoc]] ORPOConfig
