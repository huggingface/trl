# Models

With the `AutoModelForCausalLMWithValueHead` class TRL supports all decoder model architectures in transformers such as GPT-2, OPT, and GPT-Neo. In addition, with `AutoModelForSeq2SeqLMWithValueHead` you can use encoder-decoder architectures such as T5. TRL also requires reference models which are frozen copies of the model that is trained. With `create_reference_model` you can easily create a frozen copy and also share layers between the two models to save memory.


## create_reference_model

[[autodoc]] create_reference_model
