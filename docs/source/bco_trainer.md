# BCO Trainer

[![](https://img.shields.io/badge/All_models-BCO-blue)](https://huggingface.co/models?other=bco,trl)

TRL supports the Binary Classifier Optimization (BCO).
The [BCO](https://huggingface.co/papers/2404.04656) authors train a binary classifier whose logit serves as a reward so that the classifier maps {prompt, chosen completion} pairs to 1 and {prompt, rejected completion} pairs to 0.
For a full example have a look at  [`examples/scripts/bco.py`].

## Expected dataset type

The [`BCOTrainer`] requires an [unpaired preference dataset](dataset_formats#unpaired-preference).
The [`BCOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

## Expected model format
The BCO trainer expects a model of `AutoModelForCausalLM`, compared to PPO that expects `AutoModelForCausalLMWithValueHead` for the value function.

## Using the `BCOTrainer`

For a detailed example have a look at the `examples/scripts/bco.py` script. At a high level we need to initialize the `BCOTrainer` with a `model` we wish to train and a reference `ref_model` which we will use to calculate the implicit rewards of the preferred and rejected response. 

The `beta` refers to the hyperparameter of the implicit reward, and the dataset contains the 3 entries listed above. Note that the `model` and `ref_model` need to have the same architecture (ie decoder only or encoder-decoder).



```py
training_args = BCOConfig(
    beta=0.1,
)

bco_trainer = BCOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
```
After this one can then call:

```py
bco_trainer.train()
```

## Underlying Distribution matching (UDM)

In practical scenarios, the thumbs-up and thumbs-down datasets are likely to have divergent underlying distributions of prompts.
Consider an LLM deployed for user feedback: if the model excels in writing tasks but underperforms in coding, the thumbs-up dataset will be dominated by writing-related prompts, while the thumbs-down dataset will contain mostly coding-related prompts.  
If the prompts in your desired and undesired datasets differ a lot, it is useful to enable UDM.  

Choose an embedding model and tokenizer:

```py
embedding_model = AutoModel.from_pretrained(your_model_id)
embedding_tokenizer = AutoTokenizer.from_pretrained(your_model_id)

# customize this function depending on your embedding model
def embed_prompt(input_ids, attention_mask, model):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state.mean(dim=1)

embedding_model = Accelerator().prepare_model(self.embedding_model)
embedding_func = partial(embed_prompt, model=embedding_model)
```

Set `prompt_sample_size` to define how many prompts are selected to train the UDM classifier and start the training with the provided embedding function:

```py
training_args = BCOConfig(
    beta=0.1,
    prompt_sample_size=512,
)

bco_trainer = BCOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    embedding_func=embedding_func,
    embedding_tokenizer=self.embedding_tokenizer,
)

bco_trainer.train()
```

### For Mixture of Experts Models: Enabling the auxiliary loss

MOEs are the most efficient if the load is about equally distributed between experts.  
To ensure that we train MOEs similarly during preference-tuning, it is beneficial to add the auxiliary loss from the load balancer to the final loss.  

This option is enabled by setting `output_router_logits=True` in the model config (e.g. MixtralConfig).  
To scale how much the auxiliary loss contributes to the total loss, use the hyperparameter `router_aux_loss_coef=...` (default: 0.001).

## BCOTrainer

[[autodoc]] BCOTrainer

## BCOConfig

[[autodoc]] BCOConfig
