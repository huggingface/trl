"""
`CUDA_VISIBLE_DEVICES=1 python mre.py` works fine
without bnb: `CUDA_VISIBLE_DEVICES=1 python mre.py` doesn't work (diverges)
`accelerate launch mre.py` diverges

Seems to be training without bnb that fails!
"""

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, Idefics2ForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig
from trl import get_kbit_device_map

USE_QLORA = True  # QLora

if __name__ == "__main__":
    # Load the model and processor
    model_name = "HuggingFaceM4/idefics2-8b"
    if USE_QLORA:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    else:
        quantization_config = None
    model = Idefics2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$",
        init_lora_weights="gaussian",
        use_dora=False if USE_QLORA else True,
    )
    model.add_adapter(lora_config)
    model.enable_adapters()

    processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)

    # Load a dataset
    dataset = load_dataset("HuggingFaceH4/llava-instruct-mix-vsft")
    # dataset = load_dataset("HuggingFaceH4/cord-v2")

    # Process the dataset
    def data_collator(examples, add_generation_prompt=False):
        messages = [example["messages"] for example in examples]
        images = [example["images"] for example in examples]
        text = processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        batch = processor(text, images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        if processor.tokenizer.pad_token_id is not None:
            image_token = processor.tokenizer("<image>", add_special_tokens=False).input_ids[0]
            labels[labels == processor.tokenizer.pad_token_id] = image_token
        batch["labels"] = labels
        return batch

    # Test before training
    # example = dataset["test"][0]
    # example["messages"] = example["messages"][:-1]  # remove the last message (it's the answer)
    # example["images"][0].save("image.jpg")
    # inputs = data_collator([example], add_generation_prompt=True)
    # exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    # bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    # generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_new_tokens=1000)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # for i, t in enumerate(generated_text):
    #     print(f"{i}:\n{t}\n")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            logging_steps=10,
            num_train_epochs=1,
            logging_dir="./logs",
            remove_unused_columns=False,
            max_grad_norm=1.0,
        ),
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    trainer.train()

    # Save the model
    model.save_pretrained("idefics2-8b-fst-llava-instruct-mix")

    # Test after training
    # example = dataset["test"][0]
    # example["messages"] = example["messages"][:-1]  # remove the last message (it's the answer)
    # inputs = data_collator([example], add_generation_prompt=True)
    # exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    # bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    # generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_new_tokens=1000)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # for i, t in enumerate(generated_text):
    #     print(f"{i}:\n{t}\n")


# accelerate launch python sft_idefics.py
# OK

# Issues:

# python mre.py
# TypeError: DynamicCache.__init__() takes 1 positional argument but 2 were given

# python mre.py with LORA and no QLORA, diverges (all numbers of devices)
