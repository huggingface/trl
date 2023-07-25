import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertConfig,
    BertTokenizer,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    pipeline,
)

from trl import SoftQLearningTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]


def collate_fn(batch):
    # Tokenize and pad batch while also getting attention masks
    xs = [y[0] for y in batch]
    ys = [y[1] for y in batch]
    encoded_xs = tokenizer.batch_encode_plus(
        xs,
        padding="longest",  # Pad to longest sequence in batch
        truncation=True,
        max_length=30,  # Max length to avoid running out of memory
        return_attention_mask=True,  # Return attention mask
        return_length=True,  # Return length of each sequence
        return_tensors="pt",  # Return PyTorch tensors
    )

    encoded_ys = tokenizer.batch_encode_plus(
        ys,
        padding="longest",  # Pad to longest sequence in batch
        truncation=True,
        max_length=30,  # Max length to avoid running out of memory
        return_attention_mask=True,  # Return attention mask
        return_length=True,  # Return length of each sequence
        return_tensors="pt",  # Return PyTorch tensors
    )

    return (encoded_xs, encoded_ys)


class CustomTextDataset(Dataset):
    def __init__(self, data_filename, labels_filename):
        with open(data_filename, "r") as f:
            self.data = f.read().splitlines()

        with open(labels_filename, "r") as f:
            self.labels = f.read().splitlines()

        assert len(self.data) == len(self.labels), "Mismatched lengths of data and labels!"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Tokenize data and labels here, the exact method will depend on your tokenizer
        data = self.data[index]
        label = self.labels[index]

        return data, label


def compute_perplexities(
    sentences: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
) -> Tuple[FloatTensor, FloatTensor]:

    nlls = []
    for sentence in sentences:
        encodings = tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        try:
            # labels **are shifted** inside the model
            outputs = model(input_ids, labels=input_ids.clone())
            nll = outputs[0]
        except RuntimeError:
            # Could happen when the input is empty
            nll = torch.tensor(float("nan")).to(device)

        nlls.append(nll)

    stacked_nlls = torch.stack(nlls, dim=0)
    return stacked_nlls, stacked_nlls.exp()


class GPT2TopicReward(object):
    WORDLISTS_BASE_DIR = "./demo_data/wordlists"
    PPLM_INPUTS_FILE_NAME = "./demo_data/pplm-inputs.txt"
    TOPICS = ["legal", "politics", "computers", "space", "religion", "science", "military"]

    def __init__(
        self,
        max_length: int = 60,
        num_return_sequences_train: int = 2,
        num_return_sequences_infer: int = 100,
        topic_scores_aggregator: Optional[Callable[[List[float]], Union[float, np.number]]] = None,
        include_perplexity: bool = True,
        return_intermediate_outputs: bool = False,
    ) -> None:

        if topic_scores_aggregator is None:
            # Use the average by default
            topic_scores_aggregator = lambda scores: np.mean(scores)

        # if include_perplexity is True:
        #    sql_utils.colorful_warning("Adding Perplexity-based Reward", bg="blue")

        # sql_utils.colorful_warning(f"max_length={max_length}", bg="blue")

        # https://huggingface.co/gpt2
        # https://huggingface.co/facebook/bart-large-mnli
        self._generator = pipeline("text-generation", model="distilgpt2", device=0)
        self._classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

        self._max_length = max_length
        self._num_return_sequences_train = num_return_sequences_train
        self._num_return_sequences_infer = num_return_sequences_infer
        self._topic_scores_aggregator = topic_scores_aggregator
        # `topic_to_candidate_labels_map` is deprecated
        (
            self._topic_to_candidate_labels_map,
            self._pplm_inputs,
        ) = self.load_topic_to_candidate_labels_map_and_pplm_inputs()

        # Technically, adding perplexity-based reward will break
        # the scale, but we will ignore this for now since
        # this number is relatively small.
        self._include_perplexity = include_perplexity
        # Do not set is to `True` during training, use it for debugging.
        self._return_intermediate_outputs = return_intermediate_outputs

    def load_topic_to_candidate_labels_map_and_pplm_inputs(self) -> Tuple[Dict[str, List[str]], List[str]]:
        topic_to_candidate_labels_map = {}
        for topic in self.TOPICS:
            file_name = os.path.join(self.WORDLISTS_BASE_DIR, f"{topic}.txt")

            with open(file_name) as f:
                # There is one file that capitalized all words
                # hence it's likely better to lower case all of
                # them -- with the risk of hurting some words
                topic_to_candidate_labels_map[topic] = [d.strip().lower() for d in f.readlines()]

        with open(self.PPLM_INPUTS_FILE_NAME) as f:
            pplm_inputs = [d.strip() for d in f.readlines()]

        return topic_to_candidate_labels_map, pplm_inputs

    def _format_prompts(self, strings: List[str]) -> List[str]:
        inputs = np.random.choice(
            self._pplm_inputs,
            size=len(strings),
            # we use with-replacement here
            replace=True,
        ).tolist()

        new_strings = [self._generator.tokenizer.convert_tokens_to_string(s.split()) for s in strings]

        return [f"{s_1} {s_2}" for s_1, s_2 in zip(new_strings, inputs)]

    def _compute_nll_reward(self, sentences: List[str]) -> FloatTensor:
        nlls, _ = compute_perplexities(
            sentences=sentences, model=self._generator.model, tokenizer=self._generator.tokenizer
        )
        # When the sentence has just one token,
        # the NLL/perplexity will be `NaN`.
        # Further, we use the negative NLL as the reward
        return -torch.nan_to_num(nlls, nan=10.0).mean()

    def _check_classifier_outputs(
        self,
        candidate_labels: List[str],
        classifier_outputs: List[Dict],
    ) -> None:
        for output in classifier_outputs:
            if len(output["scores"]) != len(candidate_labels):
                raise ValueError

    def forward(
        self, topics: List[str], prompts: List[str], to_tensor: bool, mode: str
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        if mode not in ["train", "infer"]:
            raise ValueError

        if mode == "train":
            num_return_sequences = self._num_return_sequences_train
        if mode == "infer":
            num_return_sequences = self._num_return_sequences_infer

        # - List of length `len(prompts)`
        #     - List of length `num_return_sequences`
        #         - Dict of {"generated_text": str}
        formatted_prompts = self._format_prompts(prompts)
        print(formatted_prompts)
        generator_outputs: List[List[Dict[str, Any]]] = self._generator(
            formatted_prompts,
            max_length=self._max_length,
            num_return_sequences=num_return_sequences,
            # Only return generated text, without the prompt
            return_full_text=False,
        )

        all_classifier_outputs = []
        rewards: List[FloatTensor] = []
        quantities_to_log: Dict[str, List[FloatTensor]] = defaultdict(list)
        for batch_index in range(len(prompts)):
            generated_texts = [output["generated_text"] for output in generator_outputs[batch_index]]

            # - List of length `len(generated_texts)`
            #     - Dict of {
            #         "labels": List of length `num_topics`,
            #         "scores": List of length `num_topics`,
            #         "sequence": str,
            #     }
            try:
                topic = topics[batch_index]
                classifier_outputs = self._classifier(generated_texts, candidate_labels=[topic], multi_label=True)

                self._check_classifier_outputs(candidate_labels=[topic], classifier_outputs=classifier_outputs)

                _reward_list = [self._topic_scores_aggregator(output["scores"]) for output in classifier_outputs]

                # We assume rewards are in `[0, 100]`
                reward = torch.tensor(_reward_list).float().mean() * 100
                quantities_to_log["topic"].append(reward)
                if self._include_perplexity is True:
                    nll_reward = self._compute_nll_reward(sentences=generated_texts)
                    reward = reward + nll_reward
                    quantities_to_log["nll"].append(nll_reward)

                rewards.append(reward)
                all_classifier_outputs.append(classifier_outputs)

            except ValueError as err:
                # This happens when the generated text itself includes the
                # `</s>` token, which does happen and will cause the classifier to fail.
                # So we just ignore this error and give a score of zero for this batch.
                if str(err) != "All examples must have the same number of <eos> tokens.":
                    raise err

                rewards.append(torch.tensor(0.0).to(device))

        rewards_tensor = torch.stack(rewards)
        rewards_log = dict(
            (reward_key, torch.stack(reward_vals, dim=0).mean())
            for reward_key, reward_vals in quantities_to_log.items()
        )

        if self._return_intermediate_outputs is True:
            rewards_log["quantities_to_log"] = quantities_to_log  # type: ignore
            rewards_log["formatted_prompts"] = formatted_prompts  # type: ignore
            rewards_log["generator_outputs"] = generator_outputs  # type: ignore
            rewards_log["all_classifier_outputs"] = all_classifier_outputs  # type: ignore

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    def __call__(
        self,
        sources: List[str],
        targets: List[str],
        predictions: List[str],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], FloatTensor], Dict[str, Any]]:
        return self.forward(topics=sources, prompts=predictions, to_tensor=to_tensor, mode=mode)


source_file_name = "./demo_data/train.sources.20210424"
labels_file_name = "./demo_data/train.targets.20210424"

config_decoder, config_encoder = BertConfig(), BertConfig()
config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = EncoderDecoderModel(config=config)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

dataset = CustomTextDataset(source_file_name, labels_file_name)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)

qtrainer = SoftQLearningTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_function=GPT2TopicReward(),
    mix_strategy="alternate",
    reward_shaping=True,
    sql_loss_impl="v2_v2r_v3_v3r",
    target_model=None,
    target_update_method="polyak",
    target_learning_rate=0.001,
    device=device,
)
#
for i, batch in enumerate(dataloader):
    # qtrainer._forward_SQL(ForwardMode.SQL_OFF, batch)
    logs = qtrainer.step(batch, i)
    print(logs)