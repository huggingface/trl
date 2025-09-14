"""Test for issue #3957 - VLM KeyError fix"""

from unittest.mock import MagicMock
import torch
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling


def test_collator():
    processor = MagicMock()
    processor.apply_chat_template = MagicMock(return_value=["test"])
    processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

    collator = DataCollatorForVisionLanguageModeling(processor=processor)

    # Test with images
    examples = [{"images": ["img"], "messages": [{"role": "user", "content": "test"}]}]
    collator(examples)
    assert "images" in processor.call_args.kwargs

    # Test without images (failed before)
    processor.reset_mock()
    processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    examples = [{"messages": [{"role": "user", "content": "test"}]}]
    collator(examples)
    assert "images" not in processor.call_args.kwargs

    print("Tests passed")


if __name__ == "__main__":
    test_collator()
