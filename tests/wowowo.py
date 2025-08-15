

from unittest.mock import patch, call
from trl import BEMACallback
from trl import BEMACallback
from trl import SFTTrainer
from datasets import load_dataset


cb = BEMACallback(update_freq=3, update_after=1)
dataset = load_dataset("trl-internal-testing/zen", "standard_language_modeling")

with patch.object(cb, "_update_bema_weights") as mock_update:
    trainer = SFTTrainer(
        model="Qwen/Qwen2.5-0.5B",
        train_dataset=dataset["train"],
        callbacks=[cb],
    )
    trainer.train()

    # Check number of calls. The dataset has 17 elements, the batch size is 8 and there are 3 epochs. So there should 9 steps.
    # The BEMA callback is set to update every 2 steps, so it should be called 5 times (2, 4, 6, 8)
    assert mock_update.call_count == 4
    assert mock_update.call_args_list == [call(2), call(4), call(6), call(8)]

