from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import ModelConfig
from trl.commands.cli_utils import TrlParser
from trl.trainer.odpo import ODPOTrainer
from trl.trainer.online_dpo_config import ODPOConfig


"""
python examples/scripts/online_dpo.py --output_dir online_dpo
"""

if __name__ == "__main__":
    parser = TrlParser((ODPOConfig, ModelConfig))
    training_args, model_config = parser.parse_args_and_config()

    model = AutoModelForCausalLM.from_pretrained("cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr")
    ref_model = AutoModelForCausalLM.from_pretrained("cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr"
    )
    tokenizer = AutoTokenizer.from_pretrained("cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr", padding_side="left")

    dataset = load_dataset("trl-internal-testing/tldr-preference-sft-trl-style")

    # For simplicity, we only use the first 1024 tokens
    for split in dataset:
        dataset[split] = dataset[split].select(range(1024))

    trainer = ODPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )
    trainer.train()
