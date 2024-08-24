from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig
from transformers.integrations import WandbCallback

from trl import ModelConfig
from trl.commands.cli_utils import TrlParser
from trl.trainer.odpo import ODPOTrainer, truncate_right
from trl.trainer.online_dpo_config import ODPOConfig


class LogCompletionsCallback(WandbCallback):
    def __init__(self, prompts, freq=None):
        super().__init__()
        self.inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        self.table = []
        self._last_logged_step = -1
        self.freq = freq

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step == self._last_logged_step:
            return
        freq = self.freq or state.save_steps
        if state.global_step % freq != 0:
            return
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        model.eval()
        generation_config = GenerationConfig(
            max_new_tokens=args.completion_length, min_new_tokens=args.completion_length
        )
        inputs = self.inputs.to(args.device)
        _, context_length = inputs["input_ids"].shape
        output = model.generate(**inputs, generation_config=generation_config)
        completion_ids = output[:, context_length:]  # completions.shape[1] == self.args.completion_length
        completion_ids, _ = truncate_right(completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
        prompts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        global_step = [str(state.global_step)] * len(prompts)
        data = list(zip(global_step, prompts, completions))
        self.table.extend(data)
        table = self._wandb.Table(columns=["step", "prompt", "completion"], data=self.table)
        self._wandb.log({"completions": table})
        self._last_logged_step = state.global_step


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

    trainer = ODPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        callbacks=[LogCompletionsCallback(dataset["test"]["prompt"][:8])],
    )
    trainer.train()
