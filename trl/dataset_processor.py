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
import math
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from rich.console import Console
from rich.text import Text
from transformers import PreTrainedTokenizer


logging.basicConfig(level=logging.INFO)


COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]
# Preference dataset
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


# Chat templates
CHAT_TEMPLATES = {
    "simple_concat": "{% for message in messages %}{{' ' if not loop.first else ''}}{{message['content']}}{% endfor %}{{eos_token}}",
    "simple_chat": "{% for message in messages %}{{'\n\n' if not loop.first else ''}}{{message['role']|capitalize + ': ' +message['content']}}{% endfor %}{{eos_token}}",
    "zephyr": """{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}""",
}


@dataclass
class DatasetConfig:
    # dataset specs
    dataset_name: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_eval_split: str = "test"
    chat_template: str = "simple_concat"

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

        if self.chat_template not in CHAT_TEMPLATES:
            raise ValueError(f"chat_template must be one of {list(CHAT_TEMPLATES.keys())}")


class DatasetProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            logging.warn(
                "Tokenizer's pad token is the same as EOS token, this might cause the model to not learn to generate EOS tokens."
            )

    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        raise NotImplementedError

    def filter(self, dataset: DatasetDict):
        if self.config is None:
            logging.warn("No config provided, skipping filtering")
            return dataset
        raise NotImplementedError

    def sanity_check_(self, dataset: DatasetDict):
        """Sanity check the dataset by selecting a subset of samples to speed up tokenization: only useful for debugging"""
        if self.config.sanity_check:
            for key in dataset:
                dataset[key] = dataset[key].select(range(min(self.config.sanity_check_max_samples, len(dataset[key]))))

    def get_token_length_stats(self, features: list[str], dataset: Union[Dataset, DatasetDict]):
        """Get token length statistics for the dataset"""
        if isinstance(dataset, Dataset):
            return self._get_token_length_stats(features, dataset)
        elif isinstance(dataset, DatasetDict):
            stats = {}
            for key in dataset:
                stats[key] = self._get_token_length_stats(features, dataset[key])
            return stats

    def _get_token_length_stats(self, features: list[str], dataset: Dataset):
        stats = {}
        for key in features:
            stats[key] = {
                "max_token_length": max(len(x) for x in dataset[key]),
                "min_token_length": min(len(x) for x in dataset[key]),
                "mean_token_length": sum(len(x) for x in dataset[key]) / len(dataset[key]),
            }
        return stats

    def get_token_length_visualization(
        self, features: list[str], dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30
    ):
        """Visualize the token length distribution of the dataset"""
        num_splits = len(dataset)
        cols = min(3, num_splits)  # Maximum 3 columns
        rows = math.ceil(num_splits / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        fig.suptitle("Token Length Distribution", fontsize=16)

        for idx, (split_name, item) in enumerate(dataset.items()):
            row = idx // cols
            col = idx % cols
            ax = axs[row, col]

            for feature in features:
                token_lengths = [len(x) for x in item[feature]]
                ax.hist(token_lengths, bins=bins, alpha=0.5, label=feature, edgecolor="black")

            ax.set_title(f"{split_name} split")
            ax.set_xlabel("Token Length")
            ax.set_ylabel("Frequency")
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(f"Saved token length distribution plot to {save_path}")
        plt.close(fig)  # Close the figure to free up memory


class PreferenceDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
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

    def filter(self, dataset: Union[Dataset, DatasetDict]):
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

        filtered_dataset = dataset.filter(
            filter_fn, num_proc=self.config.num_proc, load_from_cache_file=self.config.load_from_cache_file
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_CHOSEN_KEY, INPUT_IDS_REJECTED_KEY], dataset=dataset
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_CHOSEN_KEY, INPUT_IDS_REJECTED_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class SFTDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
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

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
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
