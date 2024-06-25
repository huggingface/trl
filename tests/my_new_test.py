import datasets
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from trl import DPOConfig, DPOTrainer


# Get the model
model_id = "trl-internal-testing/tiny-random-idefics2"
model = AutoModelForVision2Seq.from_pretrained(model_id)
ref_model = AutoModelForVision2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Get the training args
training_args = DPOConfig(
    output_dir=".",
    per_device_train_batch_size=2,
    max_steps=3,
    remove_unused_columns=False,
    gradient_accumulation_steps=1,
    learning_rate=9e-1,
    evaluation_strategy="steps",
    beta=0.1,
    loss_type="sigmoid",
    precompute_ref_log_probs=True,
)

dummy_dataset_dict = {
    "images": [
        [Image.new("RGB", (100, 100), color="black")],
        [Image.new("RGB", (133, 100), color="red")],
        [Image.new("RGB", (100, 133), color="green")],
        [Image.new("RGB", (133, 133), color="blue")],
        [Image.new("RGB", (200, 50), color="yellow")],
        [Image.new("RGB", (50, 200), color="magenta")],
        [Image.new("RGB", (200, 200), color="cyan")],
        # [Image.new("RGB", (50, 50), color="white")],
        # [Image.new("RGB", (100, 100), color="orange")],
    ],
    "prompt": [
        "<image> hello",
        "<image> how are you",
        "<image> What is your name?",
        "<image> What is your name?",
        "<image> Which is the best programming language?",
        "<image> Which is the best programming language?",
        "<image> Which is the best programming language?",
        # "[INST] How is the stock price? [/INST]",
        # "[INST] How is the stock price? [/INST] ",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Python",
        # "$46 as of 10am EST",
        # "46 as of 10am EST",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "Java",
        # " $46 as of 10am EST",
        # " 46 as of 10am EST",
    ],
}

features = datasets.Features(
    {
        "images": datasets.Sequence(datasets.Image(decode=True)),  # datasets still handles badly sequence of images
        "prompt": datasets.Value("string"),
        "chosen": datasets.Value("string"),
        "rejected": datasets.Value("string"),
    }
)
dataset = Dataset.from_dict(dummy_dataset_dict, features=features)


trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    tokenizer=processor,
    train_dataset=dataset,
    eval_dataset=dataset,
)

previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

trainer.train()

assert trainer.state.log_history[-1]["train_loss"] is not None

# check the params have changed
for n, param in previous_trainable_params.items():
    new_param = trainer.model.get_parameter(n)
    # check the params have changed - ignore 0 biases
    if param.sum() != 0:
        assert not torch.allclose(param, new_param, rtol=1e-12, atol=1e-12)
