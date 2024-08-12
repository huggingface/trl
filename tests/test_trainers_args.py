import tempfile
import unittest

import datasets
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from trl import (
    BCOConfig,
    BCOTrainer,
    CPOConfig,
    CPOTrainer,
    DPOConfig,
    DPOTrainer,
    KTOConfig,
    KTOTrainer,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    ORPOConfig,
    ORPOTrainer,
    SFTConfig,
    SFTTrainer,
)


class TrainerArgTester(unittest.TestCase):
    def test_bco(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = datasets.Dataset.from_dict(
            {"prompt": ["Hello, world!"], "completion": ["This is a test completion."], "label": [True]}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = BCOConfig(
                tmp_dir,
                max_length=256,
                max_prompt_length=64,
                max_completion_length=64,
                beta=0.5,
                label_pad_token_id=-99,
                padding_value=-99,
                truncation_mode="keep_start",
                # generate_during_eval=True, # ignore this one, it requires wandb
                is_encoder_decoder=True,
                precompute_ref_log_probs=True,
                model_init_kwargs={"trust_remote_code": True},
                ref_model_init_kwargs={"trust_remote_code": True},
                dataset_num_proc=4,
                prompt_sample_size=512,
                min_density_ratio=0.2,
                max_density_ratio=20.0,
            )
            trainer = BCOTrainer(model="gpt2", ref_model="gpt2", args=args, train_dataset=dataset, tokenizer=tokenizer)
            self.assertEqual(trainer.args.max_length, 256)
            self.assertEqual(trainer.args.max_prompt_length, 64)
            self.assertEqual(trainer.args.max_completion_length, 64)
            self.assertEqual(trainer.args.beta, 0.5)
            self.assertEqual(trainer.args.label_pad_token_id, -99)
            self.assertEqual(trainer.args.padding_value, -99)
            self.assertEqual(trainer.args.truncation_mode, "keep_start")
            # self.assertEqual(trainer.args.generate_during_eval, True)
            self.assertEqual(trainer.args.is_encoder_decoder, True)
            self.assertEqual(trainer.args.precompute_ref_log_probs, True)
            self.assertEqual(trainer.args.model_init_kwargs, {"trust_remote_code": True})
            self.assertEqual(trainer.args.ref_model_init_kwargs, {"trust_remote_code": True})
            self.assertEqual(trainer.args.dataset_num_proc, 4)
            self.assertEqual(trainer.args.prompt_sample_size, 512)
            self.assertEqual(trainer.args.min_density_ratio, 0.2)
            self.assertEqual(trainer.args.max_density_ratio, 20.0)

    def test_cpo(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = datasets.Dataset.from_dict(
            {"prompt": ["Hello, world!"], "chosen": ["Nice to meet you."], "rejected": ["I don't like you."]}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = CPOConfig(
                tmp_dir,
                max_length=256,
                max_prompt_length=64,
                max_completion_length=64,
                max_target_length=64,
                beta=0.5,
                label_smoothing=0.5,
                loss_type="hinge",
                disable_dropout=False,
                cpo_alpha=0.5,
                simpo_gamma=0.2,
                label_pad_token_id=-99,
                padding_value=-99,
                truncation_mode="keep_start",
                # generate_during_eval=True, # ignore this one, it requires wandb
                is_encoder_decoder=True,
                model_init_kwargs={"trust_remote_code": True},
                dataset_num_proc=4,
            )
            trainer = CPOTrainer(model="gpt2", args=args, train_dataset=dataset, tokenizer=tokenizer)
            self.assertEqual(trainer.args.max_length, 256)
            self.assertEqual(trainer.args.max_prompt_length, 64)
            self.assertEqual(trainer.args.max_completion_length, 64)
            self.assertEqual(trainer.args.max_target_length, 64)
            self.assertEqual(trainer.args.beta, 0.5)
            self.assertEqual(trainer.args.label_smoothing, 0.5)
            self.assertEqual(trainer.args.loss_type, "hinge")
            self.assertEqual(trainer.args.disable_dropout, False)
            self.assertEqual(trainer.args.cpo_alpha, 0.5)
            self.assertEqual(trainer.args.simpo_gamma, 0.2)
            self.assertEqual(trainer.args.label_pad_token_id, -99)
            self.assertEqual(trainer.args.padding_value, -99)
            self.assertEqual(trainer.args.truncation_mode, "keep_start")
            # self.assertEqual(trainer.args.generate_during_eval, True)
            self.assertEqual(trainer.args.is_encoder_decoder, True)
            self.assertEqual(trainer.args.model_init_kwargs, {"trust_remote_code": True})
            self.assertEqual(trainer.args.dataset_num_proc, 4)

    def test_dpo(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = datasets.Dataset.from_dict(
            {"prompt": ["Hello, world!"], "chosen": ["Nice to meet you."], "rejected": ["I don't like you."]}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = DPOConfig(
                tmp_dir,
                beta=0.5,
                label_smoothing=0.5,
                loss_type="hinge",
                label_pad_token_id=-99,
                padding_value=-99,
                truncation_mode="keep_start",
                max_length=256,
                max_prompt_length=64,
                max_target_length=64,
                is_encoder_decoder=True,
                disable_dropout=False,
                # generate_during_eval=True, # ignore this one, it requires wandb
                precompute_ref_log_probs=True,
                dataset_num_proc=4,
                model_init_kwargs={"trust_remote_code": True},
                ref_model_init_kwargs={"trust_remote_code": True},
                model_adapter_name="dummy_adapter",
                ref_adapter_name="dummy_adapter",
                reference_free=True,
                force_use_ref_model=True,
                f_divergence_type="js_divergence",
                f_alpha_divergence_coef=0.5,
                sync_ref_model=True,
                ref_model_mixup_alpha=0.5,
                ref_model_sync_steps=32,
                rpo_alpha=0.5,
            )
            trainer = DPOTrainer(model="gpt2", ref_model="gpt2", args=args, train_dataset=dataset, tokenizer=tokenizer)
            self.assertEqual(trainer.args.beta, 0.5)
            self.assertEqual(trainer.args.label_smoothing, 0.5)
            self.assertEqual(trainer.args.loss_type, "hinge")
            self.assertEqual(trainer.args.label_pad_token_id, -99)
            self.assertEqual(trainer.args.padding_value, -99)
            self.assertEqual(trainer.args.truncation_mode, "keep_start")
            self.assertEqual(trainer.args.max_length, 256)
            self.assertEqual(trainer.args.max_prompt_length, 64)
            self.assertEqual(trainer.args.max_target_length, 64)
            self.assertEqual(trainer.args.is_encoder_decoder, True)
            self.assertEqual(trainer.args.disable_dropout, False)
            # self.assertEqual(trainer.args.generate_during_eval, True)
            self.assertEqual(trainer.args.precompute_ref_log_probs, True)
            self.assertEqual(trainer.args.dataset_num_proc, 4)
            self.assertEqual(trainer.args.model_init_kwargs, {"trust_remote_code": True})
            self.assertEqual(trainer.args.ref_model_init_kwargs, {"trust_remote_code": True})
            self.assertEqual(trainer.args.model_adapter_name, "dummy_adapter")
            self.assertEqual(trainer.args.ref_adapter_name, "dummy_adapter")
            self.assertEqual(trainer.args.reference_free, True)
            self.assertEqual(trainer.args.force_use_ref_model, True)
            self.assertEqual(trainer.args.f_divergence_type, "js_divergence")
            self.assertEqual(trainer.args.f_alpha_divergence_coef, 0.5)
            self.assertEqual(trainer.args.sync_ref_model, True)
            self.assertEqual(trainer.args.ref_model_mixup_alpha, 0.5)
            self.assertEqual(trainer.args.ref_model_sync_steps, 32)
            self.assertEqual(trainer.args.rpo_alpha, 0.5)

    def test_kto(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = datasets.Dataset.from_dict(
            {"prompt": ["Hello, world!"], "completion": ["This is a test completion."], "label": [True]}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = KTOConfig(
                tmp_dir,
                max_length=256,
                max_prompt_length=64,
                max_completion_length=64,
                beta=0.5,
                desirable_weight=0.5,
                undesirable_weight=0.5,
                label_pad_token_id=-99,
                padding_value=-99,
                truncation_mode="keep_start",
                # generate_during_eval=True, # ignore this one, it requires wandb
                is_encoder_decoder=True,
                precompute_ref_log_probs=True,
                model_init_kwargs={"trust_remote_code": True},
                ref_model_init_kwargs={"trust_remote_code": True},
                dataset_num_proc=4,
                loss_type="bco",
                prompt_sample_size=512,
                min_density_ratio=0.2,
                max_density_ratio=20.0,
            )
            trainer = KTOTrainer(model="gpt2", ref_model="gpt2", args=args, train_dataset=dataset, tokenizer=tokenizer)
            self.assertEqual(trainer.args.max_length, 256)
            self.assertEqual(trainer.args.max_prompt_length, 64)
            self.assertEqual(trainer.args.max_completion_length, 64)
            self.assertEqual(trainer.args.beta, 0.5)
            self.assertEqual(trainer.args.desirable_weight, 0.5)
            self.assertEqual(trainer.args.undesirable_weight, 0.5)
            self.assertEqual(trainer.args.label_pad_token_id, -99)
            self.assertEqual(trainer.args.padding_value, -99)
            self.assertEqual(trainer.args.truncation_mode, "keep_start")
            # self.assertEqual(trainer.args.generate_during_eval, True)
            self.assertEqual(trainer.args.is_encoder_decoder, True)
            self.assertEqual(trainer.args.precompute_ref_log_probs, True)
            self.assertEqual(trainer.args.model_init_kwargs, {"trust_remote_code": True})
            self.assertEqual(trainer.args.ref_model_init_kwargs, {"trust_remote_code": True})
            self.assertEqual(trainer.args.dataset_num_proc, 4)
            self.assertEqual(trainer.args.loss_type, "bco")
            self.assertEqual(trainer.args.prompt_sample_size, 512)
            self.assertEqual(trainer.args.min_density_ratio, 0.2)
            self.assertEqual(trainer.args.max_density_ratio, 20.0)

    def test_online_dpo(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = datasets.Dataset.from_dict(
            {"prompt": ["Hello, world!"], "completion": ["This is a test completion."], "label": [True]}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = OnlineDPOConfig(
                tmp_dir,
                run_name="dummy_run_name",
                sanity_check=True,
                num_mini_batches=2,
                total_episodes=100,
                local_rollout_forward_batch_size=32,
                num_sample_generations=20,
                response_length=52,
                stop_token="eos",
                stop_token_id=123,
                temperature=0.5,
                penalty_reward_value=-2,
                non_eos_penalty=True,
                sft_model_path="EleutherAI/pythia-14m",
                world_size=4,
                num_total_batches=100,
                micro_batch_size=32,
                local_batch_size=64,
                batch_size=256,
                local_mini_batch_size=8,
                mini_batch_size=32,
                exp_name="dummy_exp_name",
                reward_model_path="EleutherAI/pythia-14m",
                num_epochs=2,
                beta=0.1,
                loss_type="ipo",
                disable_dropout=False,
            )
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
            ref_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
            reward_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
            trainer = OnlineDPOTrainer(
                config=args,
                tokenizer=tokenizer,
                model=model,
                ref_model=ref_model,
                reward_model=reward_model,
                train_dataset=dataset,
            )
            self.assertEqual(trainer.args.run_name, "dummy_run_name")
            self.assertEqual(trainer.args.sanity_check, True)
            self.assertEqual(trainer.args.num_mini_batches, 2)
            self.assertEqual(trainer.args.total_episodes, 100)
            self.assertEqual(trainer.args.local_rollout_forward_batch_size, 32)
            self.assertEqual(trainer.args.num_sample_generations, 20)
            self.assertEqual(trainer.args.response_length, 52)
            self.assertEqual(trainer.args.stop_token, "eos")
            self.assertEqual(trainer.args.stop_token_id, 123)
            self.assertEqual(trainer.args.temperature, 0.5)
            self.assertEqual(trainer.args.penalty_reward_value, -2)
            self.assertEqual(trainer.args.non_eos_penalty, True)
            self.assertEqual(trainer.args.sft_model_path, "EleutherAI/pythia-14m")
            # self.assertEqual(trainer.args.world_size, 4)
            # self.assertEqual(trainer.args.num_total_batches, 100)
            # self.assertEqual(trainer.args.micro_batch_size, 32)
            # self.assertEqual(trainer.args.local_batch_size, 64)
            # self.assertEqual(trainer.args.batch_size, 256)
            self.assertEqual(trainer.args.local_mini_batch_size, 8)
            # self.assertEqual(trainer.args.mini_batch_size, 32)
            self.assertEqual(trainer.args.exp_name, "dummy_exp_name")
            self.assertEqual(trainer.args.reward_model_path, "EleutherAI/pythia-14m")
            self.assertEqual(trainer.args.num_epochs, 2)
            self.assertEqual(trainer.args.beta, 0.1)
            self.assertEqual(trainer.args.loss_type, "ipo")
            self.assertEqual(trainer.args.disable_dropout, False)

    def test_orpo(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = datasets.Dataset.from_dict(
            {"prompt": ["Hello, world!"], "chosen": ["Nice to meet you."], "rejected": ["I don't like you."]}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = ORPOConfig(
                tmp_dir,
                max_length=256,
                max_prompt_length=64,
                max_completion_length=64,
                beta=0.5,
                disable_dropout=False,
                label_pad_token_id=-99,
                padding_value=-99,
                truncation_mode="keep_start",
                # generate_during_eval=True, # ignore this one, it requires wandb
                is_encoder_decoder=True,
                model_init_kwargs={"trust_remote_code": True},
                dataset_num_proc=4,
            )

            trainer = ORPOTrainer(model="gpt2", args=args, train_dataset=dataset, tokenizer=tokenizer)
            self.assertEqual(trainer.args.max_length, 256)
            self.assertEqual(trainer.args.max_prompt_length, 64)
            self.assertEqual(trainer.args.max_completion_length, 64)
            self.assertEqual(trainer.args.beta, 0.5)
            self.assertEqual(trainer.args.disable_dropout, False)
            self.assertEqual(trainer.args.label_pad_token_id, -99)

    def test_sft(self):
        dataset = datasets.Dataset.from_dict(
            {"prompt": ["Hello, world!"], "chosen": ["Nice to meet you."], "rejected": ["I don't like you."]}
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SFTConfig(
                tmp_dir,
                dataset_text_field="dummy_text_field",
                packing=True,
                max_seq_length=256,
                dataset_num_proc=4,
                dataset_batch_size=512,
                neftune_noise_alpha=0.1,
                model_init_kwargs={"trust_remote_code": True},
                dataset_kwargs={"append_concat_token": True, "skip_prepare_dataset": True},
                eval_packing=True,
                num_of_sequences=32,
                chars_per_token=4.2,
            )
            trainer = SFTTrainer("gpt2", args=args, train_dataset=dataset)
            self.assertEqual(trainer.args.dataset_text_field, "dummy_text_field")
            self.assertEqual(trainer.args.packing, True)
            self.assertEqual(trainer.args.max_seq_length, 256)
            self.assertEqual(trainer.args.dataset_num_proc, 4)
            self.assertEqual(trainer.args.dataset_batch_size, 512)
            self.assertEqual(trainer.args.neftune_noise_alpha, 0.1)
            self.assertEqual(trainer.args.model_init_kwargs, {"trust_remote_code": True})
            self.assertIn("append_concat_token", trainer.args.dataset_kwargs)
            self.assertEqual(trainer.args.dataset_kwargs["append_concat_token"], True)
            self.assertEqual(trainer.args.eval_packing, True)
            self.assertEqual(trainer.args.num_of_sequences, 32)
            self.assertEqual(trainer.args.chars_per_token, 4.2)
