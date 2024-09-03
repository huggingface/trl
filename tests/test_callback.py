import tempfile

from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainingArguments

from trl import BasePairwiseJudge, WinRateCallback


class ThreeQuatersPairwiseJudge(BasePairwiseJudge):
    """Naive pairwise judge that always returns [1, 0, 1, 1, 0, 1, 1, 1]"""

    def judge(self, prompts, completions, shuffle_order=True):
        # just check that the batch size is 4
        assert len(prompts) == 8
        return [1, 0, 1, 1, 0, 1, 1, 1]


class TrainerWithRefModel(Trainer):
    # This is a dummy class to test the callback. Compared to the Trainer class, it only has an additional
    # ref_model attribute
    def __init__(self, model, ref_model, args, trainer_dataset, eval_dataset, tokenizer):
        super().__init__(
            model=model, args=args, train_dataset=trainer_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer
        )
        self.ref_model = ref_model


def test_trainer_callback():
    model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
    ref_model = AutoModelForCausalLM.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
    tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/dummy-GPT2-correct-vocab")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "prompt": [
                        "Hello world!",
                        "This is a test.",
                        "We are creating a dataset.",
                        "It has eight lines.",
                        "Each line is a sentence.",
                        "The sentences are simple.",
                        "This is just for testing.",
                        "Goodbye!",
                    ]
                }
            ),
            "test": Dataset.from_dict(
                {
                    "prompt": [
                        "The sun sets in the west.",
                        "Mountains are majestic.",
                        "Rivers flow endlessly.",
                        "Forests are full of life.",
                        "Birds sing in the morning.",
                        "Waves crash on the shore.",
                        "The moon glows at night.",
                        "Stars twinkle in the sky.",
                    ]
                }
            ),
        }
    )

    def tokenize_function(examples):
        out = tokenizer(examples["prompt"], padding="max_length", max_length=16, truncation=True)
        out["labels"] = out["input_ids"].copy()
        return out

    dataset = dataset.map(tokenize_function, batched=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = TrainingArguments(
            output_dir=tmp_dir,
            eval_strategy="steps",
            eval_steps=2,  # evaluate every 2 steps
            per_device_train_batch_size=2,  # 8 samples in total so 4 batches of 2 per epoch
            per_device_eval_batch_size=2,
            report_to="none",
        )
        trainer = TrainerWithRefModel(
            model=model,
            ref_model=ref_model,
            args=args,
            trainer_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
        )
        generation_config = GenerationConfig(max_length=32)
        win_rate_callback = WinRateCallback(
            judge=ThreeQuatersPairwiseJudge(), trainer=trainer, generation_config=generation_config
        )
        trainer.add_callback(win_rate_callback)
        trainer.train()
        winrate_history = [h for h in trainer.state.log_history if "eval_win_rate" in h]
        assert winrate_history == [
            {"eval_win_rate": 0.75, "epoch": 0.5, "step": 2},
            {"eval_win_rate": 0.75, "epoch": 1.0, "step": 4},
            {"eval_win_rate": 0.75, "epoch": 1.5, "step": 6},
            {"eval_win_rate": 0.75, "epoch": 2.0, "step": 8},
            {"eval_win_rate": 0.75, "epoch": 2.5, "step": 10},
            {"eval_win_rate": 0.75, "epoch": 3.0, "step": 12},
        ]
