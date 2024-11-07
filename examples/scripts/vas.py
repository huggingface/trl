"""
python vas.py \
    --log_with=wandb
    --ref_model_name hanseungwook/vas-llama-2-7b-hh-sft
    --model_name hanseungwook/vas-tiny-llama-1.1b-hh-sft
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, VASConfig, VASTrainer, set_seed


tqdm.pandas()


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

    # Generation kwargs
    generation_batch_size: Optional[int] = field(default=16, metadata={"help": "The batch size for generation"})
    temperature: Optional[float] = field(default=1.0, metadata={"help": "The temperature for generation"})
    top_k: Optional[float] = field(default=0.0, metadata={"help": "The top_k for generation"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "The top_p for generation"})


parser = HfArgumentParser((ScriptArguments, VASConfig))
args, vas_config = parser.parse_args_into_dataclasses()

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


def build_response_train_dataset(config, dataset_name="Anthropic/hh-rlhf"):
    ds = load_dataset(dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.unk_token_id

    def tokenize(sample):
        query = sample["chosen"][: sample["chosen"].rfind("Assistant:") + len("Assistant:")].replace("\n", " ").strip()
        sample["query"] = tokenizer.encode(query)
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


dataset = build_response_train_dataset(vas_config)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(vas_config.seed)

# Now let's build the model, the reference model, and the tokenizer.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = trl_model_class.from_pretrained(
    vas_config.model_name,
    quantization_config=quantization_config,
    trust_remote_code=args.trust_remote_code,
    device_map="auto",
)

if args.use_peft:
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.pretrained_model = prepare_model_for_kbit_training(model.pretrained_model, use_gradient_checkpointing=True)
    model.pretrained_model = get_peft_model(model.pretrained_model, peft_config)
    model.is_peft_model = True

# Initialize the value head with zeros leads to better performance
torch.nn.init.zeros_(model.v_head.summary.weight)
torch.nn.init.zeros_(model.v_head.summary.bias)

# Disable dropout
for module in model.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0

ref_model = trl_model_class.from_pretrained(
    vas_config.ref_model_name,
    quantization_config=quantization_config,
    trust_remote_code=args.trust_remote_code,
    device_map="auto",
)

tokenizer = ref_tokenizer = AutoTokenizer.from_pretrained(vas_config.model_name)

# Some tokenizers like don't have a padding token by default, so we set one here.
tokenizer.pad_token_id = tokenizer.unk_token_id
tokenizer.add_eos_token = True

# We then build the VASTrainer, passing the model, the reference model, the tokenizer
vas_trainer = VASTrainer(vas_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

device = vas_trainer.accelerator.device
if vas_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(vas_trainer.accelerator.device)
reward_model = vas_trainer.accelerator.prepare(reward_model)
reward_model.requires_grad_(False)
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
reward_model.eval()

generation_kwargs = {
    "top_k": args.top_k,
    "top_p": args.top_p,
    "temperature": args.temperature,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 100,
}

for _epoch, batch in tqdm(enumerate(vas_trainer.dataloader)):
    query_tensors = batch["query"]
    response_tensors = vas_trainer.generate(
        query_tensors, batch_size=args.generation_batch_size, return_prompt=False, **generation_kwargs
    )

    # Compute score
    full_responses = [torch.cat([query, response]) for query, response in zip(query_tensors, response_tensors)]
    texts = tokenizer.batch_decode(full_responses, skip_special_tokens=True)
    rewards = []
    for text in texts:
        inputs_ids = reward_tokenizer.encode(text, return_tensors="pt").to(reward_model.device)
        reward_outputs = reward_model(inputs_ids)
        reward = reward_outputs.logits[0]
        rewards.append(reward.squeeze())

    # Run VAS step
    stats = vas_trainer.step(query_tensors, response_tensors, rewards)
    vas_trainer.log_stats(stats, batch, rewards, columns_to_log=["query"])

vas_trainer.save_pretrained("/data/pulkitag/models/idanshen/trl/example")

# Decoding example
# query = "Human: How are you doing today? Assistant:"
# inputs = ref_tokenizer.encode(query, return_tensors='pt').to(reward_model.device)
# output = vas_trainer.generate(inputs, vas_generation=True, beta=3.0)
