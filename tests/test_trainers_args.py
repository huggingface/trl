import tempfile
import unittest

import datasets
from transformers import AutoModelForCausalLM

from trl import SFTConfig, SFTTrainer


class SFTTrainerArgTester(unittest.TestCase):
    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        # [{"role": str, "content": str}]
        self.dataset = datasets.Dataset.from_dict(
            {"messages": [[{"role": "user", "content": "hello"}], [{"role": "user", "content": "world"}]]}
        )

    # dataset_text_field: Optional[str] = None
    def test_dataset_text_field(self):
        dataset = datasets.Dataset.from_dict({"dummy_text_field": ["hello", "world"]})
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(tmp_dir, dataset_text_field="dummy_text_field")
            trainer = SFTTrainer(self.model, args=args, train_dataset=dataset)
            self.assertEqual(trainer.args.dataset_text_field, "dummy_text_field")

    # packing: Optional[bool] = False
    def test_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # we skip prepare_dataset because here we just want to test the packing argument
            args = SFTConfig(tmp_dir, packing=True, dataset_kwargs={"skip_prepare_dataset": True})
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.packing, True)

    # max_seq_length: Optional[int] = None
    def test_max_seq_length(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(tmp_dir, max_seq_length=256)
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.max_seq_length, 256)

    # dataset_num_proc: Optional[int] = None
    def test_dataset_num_proc(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(tmp_dir, dataset_num_proc=4)
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.dataset_num_proc, 4)

    # dataset_batch_size: int = 1000
    def test_dataset_batch_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(tmp_dir, dataset_batch_size=512)
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.dataset_batch_size, 512)

    # neftune_noise_alpha: Optional[float] = None
    def test_neftune_noise_alpha(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(tmp_dir, neftune_noise_alpha=0.1)
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.neftune_noise_alpha, 0.1)

    # model_init_kwargs: Optional[Dict] = None
    def test_model_init_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # the kwargs key must be in AutoModelForCausalLM args
            args = SFTConfig(tmp_dir, model_init_kwargs={"trust_remote_code": True})
            trainer = SFTTrainer("gpt2", args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.model_init_kwargs, {"trust_remote_code": True})

    # dataset_kwargs: Optional[Dict] = None
    def test_dataset_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                tmp_dir, dataset_kwargs={"append_concat_token": True}
            )  # the kwargs key must be in _prepare_dataset args
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertIn("append_concat_token", trainer.args.dataset_kwargs)
            self.assertEqual(trainer.args.dataset_kwargs["append_concat_token"], True)

    # eval_packing: Optional[bool] = None
    def test_eval_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(tmp_dir, eval_packing=True)
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.eval_packing, True)

    # num_of_sequences: Optional[int] = 1024
    def test_num_of_sequences(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(tmp_dir, num_of_sequences=32)
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.num_of_sequences, 32)

    # chars_per_token: Optional[float] = 3.6
    def test_chars_per_token(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(tmp_dir, chars_per_token=4.2)
            trainer = SFTTrainer(self.model, args=args, train_dataset=self.dataset)
            self.assertEqual(trainer.args.chars_per_token, 4.2)
