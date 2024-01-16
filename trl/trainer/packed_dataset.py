# sourced from https://github.com/MeetKai/functionary/tree/main/functionary/train/packing

from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset


def pack_data_points_by_length(lengths: List[int], max_length: int) -> List[List[int]]:
    """Pack data points into groups (each group is a new data point), will be used by PackedDataset, to reduce number of data points in training.
    Given lengths of data points, we pack them into groups such that the sum of lengths
    in each group is less than max_length. Each group will be considered as a data point (packed data point)
    This is known as: https://en.wikipedia.org/wiki/Bin_packing_problem
    There are many algorithms to implement this, but here we use the simple algorithm.
    We will pack/merge a consecutive list of data points until reaching the max_length
    Args:
        lengths (List[int]): _description_
        max_length (int): _description_

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    groups = []
    current_packed_length = 0
    current_group = []
    for i in range(len(lengths)):
        cur_length = lengths[i]
        if cur_length + current_packed_length <= max_length:
            current_packed_length += lengths[i]
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
            current_packed_length = cur_length
    if len(current_group) > 0:
        groups.append(current_group)
    return groups


def pack_data_points_FA(
    data_points: List[Dict], tokenizer: Any, pack_length: int
) -> Dict:
    """_summary_

    Args:
        data_points (List[Dict]): List of data_points to pack, each is: {"input_ids": xxx, "labels": xxx}
        tokenizer (Any): Tokenizer
        pack_length (int): packing length

    Returns:
        Dict: _description_
    """
    input_ids = []
    lengths = []
    label_ids = []
    attention_mask = []

    for index, item in enumerate(data_points):
        input_ids += item["input_ids"]
        # assert item["labels"][0] == -100 # This is to make sure that the first token won't be included in computing loss
        labels = list(item["labels"])
        labels[0] = -100
        label_ids += labels
        lengths.append(len(item["input_ids"]))
        attention_mask += [index + 1 for _ in range(len(item["input_ids"]))]

    pad_leng = pack_length - len(input_ids)  # padding to model_max_length
    if tokenizer.padding_side == "right":
        input_ids = input_ids + [tokenizer.pad_token_id for _ in range(pad_leng)]
        label_ids = label_ids + [-100 for _ in range(pad_leng)]
        attention_mask = attention_mask + [0 for _ in range(pad_leng)]
    else:
        input_ids = [tokenizer.pad_token_id for _ in range(pad_leng)] + input_ids
        label_ids = [-100 for _ in range(pad_leng)] + label_ids
        attention_mask = [0 for _ in range(pad_leng)] + attention_mask

    assert len(input_ids) == len(label_ids) == len(attention_mask) == pack_length
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(label_ids),
        "attention_mask": torch.tensor(
            attention_mask
        ),  # unsqueeze <-- because the shape is: B x 1 x N x N
    }


class PackedDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Any, pack_length: int) -> None:
        super().__init__()
        self.pack_length = pack_length
        self.tokenizer = tokenizer

        self.lengths = []
        self.data_points = []
        size = len(dataset)

        for i in range(size):
            data_point = dataset[i]
            input_length = torch.sum(data_point["attention_mask"]).item()
            n_data_point = {}
            n_data_point["input_ids"] = (
                data_point["input_ids"][:input_length]
                if tokenizer.padding_side == "right"
                else data_point["input_ids"][-input_length:]
            )

            if "labels" not in data_point:  # create labels if not existed
                labels = n_data_point["input_ids"].clone()
                labels[labels == tokenizer.pad_token_id] = -100  # mask pad_token
                n_data_point["labels"] = labels.tolist()
            else:
                n_data_point["labels"] = (
                    data_point["labels"][:input_length]
                    if tokenizer.padding_side == "right"
                    else data_point["labels"][-input_length:]
                )

            self.data_points.append(n_data_point)
            self.lengths.append(input_length)

        max_input_length = max(self.lengths)
        assert self.pack_length >= max(
            self.lengths
        ), f"pack_length must be >= max(input lengths), found pack_length={self.pack_length}, max_input_length={max_input_length}"
        self.groups = pack_data_points_by_length(self.lengths, self.pack_length)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        group = self.groups[i]
        group_data_points = [self.data_points[index] for index in group]
        return pack_data_points_FA(group_data_points, self.tokenizer, self.pack_length)

    def stat(self):
        print(
            f"number of original data points:{len(self.data_points)}; packed to: {len(self.groups)} data points"
        )
        original_avg_length = sum(self.lengths) / len(self.lengths)

        packed_lengths = []
        for group in self.groups:
            lengths = [self.lengths[index] for index in group]
            packed_lengths.append(sum(lengths))

        avg_packed_length = sum(packed_lengths) / len(packed_lengths)
        print(
            f"original avg length: {original_avg_length}; avg packed length: {avg_packed_length}"
        )
