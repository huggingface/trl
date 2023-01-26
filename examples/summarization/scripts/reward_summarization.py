from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, PreTrainedTokenizerBase, HfArgumentParser
from transformers.utils import PaddingStrategy
from typing import Optional, Union, List, Dict, Any
import evaluate
from dataclasses import dataclass, field
import torch.nn as nn
import torch
import numpy as np

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=0, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "If you want to resume training where it left off."})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."})
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[int] = field(default=2e-5)
    gpt_model_name: Optional[str] = field(default="gpt2", metadata={"help": "The gpt model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, etc."})
    bf16: Optional[bool] = field(default=False, metadata={"help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the human comparisons dataset for tuning the reward model.
ds = load_dataset("openai/summarize_from_feedback", name="comparisons")

# Define the reward model. Code here is loosely inspired by https://wandb.ai/carperai/summarize_RLHF/reports/Implementing-RLHF-Learning-to-Summarize-with-trlX--VmlldzozMzAwODM2.
# Note that input_ids_j and attention_mask_j are expected to come from the post + the summary that humans don't prefer, and input_ids_k and attention_mask_k are expected to come
# from the post + summary that humans don't prefer. If you don't specify input_ids_k and attention_mask_k, then it is assumed that you just want the reward for input_ids_j and attention_mask_j
# (this is the case after you have already trained the reward model and want it to give rewards for new data).
class GPTRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(config)
        self.config = model.config
        # gpt-neo models have hidden_size instead of n_embd.
        self.config.n_embd = (
            self.config.hidden_size
            if hasattr(self.config, "hidden_size")
            else self.config.n_embd
        )
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1)

    def forward(
        self,
        input_ids_j=None,
        attention_mask_j=None,
        input_ids_k=None,
        attention_mask_k=None,
        return_loss=True,
    ):
        transformer_outputs_j = self.transformer(
            input_ids_j,
            attention_mask=attention_mask_j,
        )
        hidden_states_j = transformer_outputs_j[0][:,-1,:]  # Get the last hidden state
        rewards_j = self.v_head(hidden_states_j)
        if input_ids_k is not None:
            transformer_outputs_k = self.transformer(
                input_ids_k,
                attention_mask=attention_mask_k,
            )
            hidden_states_k = transformer_outputs_k[0][:,-1,:]  # Get the last hidden state
            rewards_k = self.v_head(hidden_states_k)

            loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
            return {"loss": loss, "rewards_j": rewards_j, "rewards_k": rewards_k}
        return {"rewards_j": rewards_j}

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir=f"{script_args.gpt_model_name}_summarization_reward_model",
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=5,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    load_best_model_at_end=True,
    push_to_hub=True,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
)

# Load the model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.gpt_model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPTRewardModel(script_args.gpt_model_name)

# Turn the dataset into our special j (preferred summary) and k (not preferred summary) format.
def turn_into_text_classification_format(examples):
    
    new_examples = {"text_j": [], "text_k": []}

    for info, summaries, choice in zip(examples["info"], examples["summaries"], examples["choice"]):
        assert len(summaries) == 2
        assert choice in (0,1)
        original_text_field = "post" if info["post"] is not None else "article"
        new_examples["text_j"].append(summaries[choice]["text"] + " " + tokenizer.bos_token + " " + info[original_text_field])
        new_examples["text_k"].append(summaries[0 if choice==1 else 1]["text"] + " " + tokenizer.bos_token + " " + info[original_text_field])

    return new_examples

num_proc = 100 # Can adjust to be higher if you have more processors.
original_columns = ds["train"].column_names
ds = ds.map(turn_into_text_classification_format, batched=True, num_proc=num_proc, remove_columns=original_columns)

# Tokenize the dataset.
def preprocess_function(examples):
    tokenized_j = tokenizer(examples["text_j"], truncation=True)
    tokenized_k = tokenizer(examples["text_k"], truncation=True)
    return {"input_ids_j": tokenized_j["input_ids"], "attention_mask_j": tokenized_j["attention_mask"], "input_ids_k": tokenized_k["input_ids"], "attention_mask_k": tokenized_k["attention_mask"]}

tokenized_ds = ds.map(preprocess_function, batched=True, num_proc=num_proc)

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]})
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {"input_ids_j": batch_j["input_ids"], "attention_mask_j": batch_j["attention_mask"], "input_ids_k": batch_k["input_ids"], "attention_mask_k": batch_k["attention_mask"]}
        return batch

# Define the metric that we'll use for validation
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

# Train the model, woohoo.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train(script_args.resume_from_checkpoint)

# Push to the hub so you can share it with people :D
trainer.push_to_hub()
