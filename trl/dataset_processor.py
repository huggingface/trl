# this file deals with dataset pre-processing before training

# 1. PPO (prompt)
# 2. SFT (prompt + demonstration), there is also packing.
# 3. ✅ RM / DPO (chosen and rejected)
# 4. ✅ Visualization of length distributions?
# 5. ✅ Filter?
#   * Smart truncation?
# 6. ✅ dataset_num_proc
# 7. ✅ check EOS token
# 8. dataset mixer?
# 9. ✅ pretty print that show tokenization?
# 10. ✅ hashable tokneization?
# 11. inputs / labels / attention_mask
# 12. ✅ always set a `tokenizer.pad_token_id`?
# 13. a new DataCollatorForLanguageModeling?
# 14. `add_bos_token` and `add_eos_token`? E.g., LLAMA models
# 15. generate properties: has eos_token, bos_token?

## too many names related to "maximum length":
# * `max_seq_length` in SFT
# * `max_length`, `max_target_length` in RM / DPO,
# * `max_prompt_length` in DPO

import logging
import multiprocessing
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from rich.console import Console
from rich.text import Text
from transformers import PreTrainedTokenizer


COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]
# preference dataset
CHOSEN_KEY = "chosen"
REJECTED_KEY = "rejected"
INPUT_IDS_CHOSEN_KEY = "input_ids_chosen"
ATTENTION_MASK_CHOSEN_KEY = "attention_mask_chosen"
INPUT_IDS_REJECTED_KEY = "input_ids_rejected"
ATTENTION_MASK_REJECTED_KEY = "attention_mask_rejected"
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"

# SFT dataset
MESSAGES_KEY = "messages"
INPUT_IDS_KEY = "input_ids"


@dataclass
class DatasetConfig:
    max_token_length: Optional[int] = None
    max_prompt_token_lenth: Optional[int] = None

    # dataset.map config
    sanity_check: bool = False
    sanity_check_max_samples: int = 100
    batched: bool = False
    load_from_cache_file: Optional[bool] = None
    num_proc: Optional[int] = None

    # visualization configs
    ncols: int = 2

    def __post_init__(self):
        if self.sanity_check:
            self.num_proc = 1
            self.load_from_cache_file = False
        else:
            self.num_proc = multiprocessing.cpu_count()
            self.load_from_cache_file = True


class DatasetProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            logging.warn(
                "Tokenizer's pad token is the same as EOS token, this might cause the model to not learn to generate EOS tokens."
            )

    def tokenize(self, dataset: Dataset):
        raise NotImplementedError

    def filter(self, dataset: Dataset):
        if self.config is None:
            logging.warn("No config provided, skipping filtering")
            return dataset
        raise NotImplementedError

    def sanity_check_(self, dataset: DatasetDict):
        if self.config.sanity_check:
            for key in dataset:
                dataset[key] = dataset[key].select(range(min(self.sanity_check_max_samples, len(dataset[key]))))

    def get_token_length_stats(self, features: list[str], dataset: Dataset):
        stats = {}
        for key in features:
            stats[key] = {
                "max_token_length": max(len(x) for x in dataset[key]),
                "min_token_length": min(len(x) for x in dataset[key]),
                "mean_token_length": sum(len(x) for x in dataset[key]) / len(dataset[key]),
            }
        return stats

    def get_token_length_visualization(
        self, features: list[str], dataset: Dataset, save_path: str = "tmp.png", bins: int = 30
    ):
        plt.figure(figsize=(10, 5))

        for feature in features:
            token_lengths = [len(x) for x in dataset[feature]]

            # Plot the histogram of token lengths
            plt.hist(token_lengths, bins=bins, alpha=0.5, label=feature, edgecolor="black")

        # Add title and labels
        plt.title("Token Length Distribution")
        plt.xlabel("Token Length")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")
        # Show the plot
        plt.savefig(save_path)
        logging.info(f"Saved token length distribution plot to {save_path}")


class PreferenceDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(row[CHOSEN_KEY][:-1])
            row[ATTENTION_MASK_PROMPT_KEY] = [1] * len(row[INPUT_IDS_PROMPT_KEY])
            row[INPUT_IDS_CHOSEN_KEY] = self.tokenizer.apply_chat_template(row[CHOSEN_KEY])
            row[ATTENTION_MASK_CHOSEN_KEY] = [1] * len(row[INPUT_IDS_CHOSEN_KEY])
            row[INPUT_IDS_REJECTED_KEY] = self.tokenizer.apply_chat_template(row[REJECTED_KEY])
            row[ATTENTION_MASK_REJECTED_KEY] = [1] * len(row[INPUT_IDS_REJECTED_KEY])
            return row

        return dataset.map(
            tokenize_fn, num_proc=self.config.num_proc, load_from_cache_file=self.config.load_from_cache_file
        )

    def filter(self, dataset: Dataset):
        def filter_fn(row):
            return (
                len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_lenth
                if self.config.max_prompt_token_lenth is not None
                else True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
                if self.config.max_token_length is not None
                else True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
                if self.config.max_token_length is not None
                else True
            )

        return dataset.filter(
            filter_fn, num_proc=self.config.num_proc, load_from_cache_file=self.config.load_from_cache_file
        )

    def get_token_length_stats(self, dataset: Dataset):
        return super().get_token_length_stats(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_CHOSEN_KEY, INPUT_IDS_REJECTED_KEY], dataset=dataset
        )

    def get_token_length_visualization(self, dataset: Dataset, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_CHOSEN_KEY, INPUT_IDS_REJECTED_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class SFTDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(row[MESSAGES_KEY][:-1])
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row[MESSAGES_KEY])
            return row

        return dataset.map(
            tokenize_fn, num_proc=self.config.num_proc, load_from_cache_file=self.config.load_from_cache_file
        )

    def filter(self, dataset: Dataset):
        def filter_fn(row):
            return (
                len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_lenth
                if self.config.max_prompt_token_lenth is not None
                else True and len(row[INPUT_IDS_KEY]) <= self.config.max_token_length
                if self.config.max_token_length is not None
                else True
            )

        return dataset.filter(
            filter_fn, num_proc=self.config.num_proc, load_from_cache_file=self.config.load_from_cache_file
        )

    def get_token_length_stats(self, dataset: Dataset):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: Dataset, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


def visualize_token(tokens: list[int], tokenizer: PreTrainedTokenizer):
    i = 0
    console = Console()
    rich_text = Text()
    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        decoded_token = tokenizer.decode(token)
        rich_text.append(f"{decoded_token}", style=color)
    console.print(rich_text)
