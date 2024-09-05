import unittest

from datasets import Dataset, DatasetDict

from trl.data_utils import is_conversational


class TestIsConversational(unittest.TestCase):
    def test_non_preference(self):
        # Non-preference conversational dataset (prompt + completion)
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "What color is the sky?"}]],
                "completion": [[{"role": "assistant", "content": "It is blue."}]],
            }
        )
        self.assertTrue(is_conversational(dataset))

    def test_prompt_only(self):
        # Prompt-only conversational dataset (only prompt)
        dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "What color is the sky?"}]]})
        self.assertTrue(is_conversational(dataset))

    def test_preference(self):
        # Preference conversational dataset (prompt + chosen + rejected)
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "What color is the sky?"}]],
                "chosen": [[{"role": "assistant", "content": "It is blue."}]],
                "rejected": [[{"role": "assistant", "content": "It is green."}]],
            }
        )
        self.assertTrue(is_conversational(dataset))

    def test_unpaired_preference(self):
        # Unpaired preference conversational dataset (prompt + completion + label)
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": "What color is the sky?"}]],
                "completion": [[{"role": "assistant", "content": "It is green."}]],
                "label": [False],
            }
        )
        self.assertTrue(is_conversational(dataset))

    def test_non_conversational(self):
        # Non-conversational dataset (e.g., plain text without roles)
        dataset = Dataset.from_dict({"prompt": ["The sky is"]})
        self.assertFalse(is_conversational(dataset))

    def test_conversational_datasetdict(self):
        # DatasetDict with a conversational dataset (prompt + completion)
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "prompt": [[{"role": "user", "content": "What color is the sky?"}]],
                        "completion": [[{"role": "assistant", "content": "It is blue."}]],
                    }
                )
            }
        )
        self.assertTrue(is_conversational(dataset_dict))

    def test_non_conversational_datasetdict(self):
        # DatasetDict with a non-conversational dataset
        dataset_dict = DatasetDict({"train": Dataset.from_dict({"prompt": ["The sky is"]})})
        self.assertFalse(is_conversational(dataset_dict))


if __name__ == "__main__":
    unittest.main()
