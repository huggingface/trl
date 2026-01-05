import unittest
import torch
from trl.experimental.ppo.ppo_config import PPOConfig
from trl.experimental.ppo.ppo_trainer import PPOTrainer

class TestRewardShaping(unittest.TestCase):
    def setUp(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        
    def test_length_penalty(self):
        args = PPOConfig(
            reward_shaping_enable_length=True,
            reward_shaping_length_coef=0.1
        )
        
        # sequence_lengths_p1 (lengths)
        sequence_lengths = torch.tensor([10, 20], dtype=torch.long)
        # responses not used for length penalty
        responses = torch.zeros(2, 20, dtype=torch.long)
        
        delta = PPOTrainer._compute_reward_shaping_delta(responses, sequence_lengths, self.eos_token_id, args)
        
        expected = -0.1 * sequence_lengths.float()
        torch.testing.assert_close(delta, expected)

    def test_repetition_penalty(self):
        args = PPOConfig(
            reward_shaping_enable_repetition=True,
            reward_shaping_repetition_coef=1.0,
            reward_shaping_repetition_ngram=2
        )
        
        # Case 1: No repetition
        # [1, 2, 3, 4] -> ngrams: (1,2), (2,3), (3,4). Total 3, Unique 3. Rate 0.
        
        # Case 2: Full repetition
        # [1, 2, 1, 2] -> ngrams: (1,2), (2,1), (1,2). Total 3. Unique 2 {(1,2), (2,1)}. Rate 1 - 2/3 = 1/3.
        
        # Case 3: Short sequence (< n)
        # [1] -> n=2. Total 0. Rate 0.
        
        responses = torch.tensor([
            [1, 2, 3, 4],
            [1, 2, 1, 2],
            [1, 0, 0, 0]
        ], dtype=torch.long)
        sequence_lengths = torch.tensor([4, 4, 1], dtype=torch.long) # lengths
        
        delta = PPOTrainer._compute_reward_shaping_delta(responses, sequence_lengths, self.eos_token_id, args)
        
        expected = torch.tensor([
            -1.0 * 0.0,
            -1.0 * (1.0 - 2.0/3.0),
            -1.0 * 0.0
        ])
        
        torch.testing.assert_close(delta, expected)

    def test_eos_shaping(self):
        args = PPOConfig(
            reward_shaping_enable_eos=True,
            reward_shaping_eos_bonus=1.0,
            reward_shaping_eos_missing_penalty=2.0,
            reward_shaping_eos_out_of_range_penalty=0.5,
            reward_shaping_eos_min_len=2,
            reward_shaping_eos_max_len=3
        )
        
        # Case 1: Missing EOS
        # Case 2: EOS in range (len=2) -> index 1
        # Case 3: EOS in range (len=3) -> index 2
        # Case 4: EOS out of range (len=1) -> index 0 (too short)
        # Case 5: EOS out of range (len=4) -> index 3 (too long)
        
        eos = self.eos_token_id
        responses = torch.tensor([
            [1, 1, 1, 1], # Missing
            [1, eos, 0, 0], # len=2 (index 1)
            [1, 1, eos, 0], # len=3 (index 2)
            [eos, 0, 0, 0], # len=1 (index 0)
            [1, 1, 1, eos]  # len=4 (index 3)
        ], dtype=torch.long)
        
        # sequence_lengths is not used for EOS logic in my implementation, 
        # it relies on first_true_indices of EOS in responses.
        # But I still need to pass it.
        sequence_lengths = torch.tensor([4, 2, 3, 1, 4], dtype=torch.long)
        
        delta = PPOTrainer._compute_reward_shaping_delta(responses, sequence_lengths, eos, args)
        
        expected = torch.tensor([
            -2.0, # Missing
            +1.0, # Bonus
            +1.0, # Bonus
            -0.5, # Out of range (short)
            -0.5  # Out of range (long)
        ])
        
        torch.testing.assert_close(delta, expected)

    def test_combined_shaping(self):
        args = PPOConfig(
            reward_shaping_enable_length=True,
            reward_shaping_length_coef=0.1,
            reward_shaping_enable_eos=True,
            reward_shaping_eos_bonus=1.0,
            reward_shaping_eos_missing_penalty=0.0,
            reward_shaping_eos_out_of_range_penalty=0.0,
            reward_shaping_eos_min_len=1,
            reward_shaping_eos_max_len=10
        )
        
        # [1, eos] -> len 2.
        # Length penalty: -0.1 * 2 = -0.2
        # EOS bonus: +1.0
        # Total: 0.8
        
        responses = torch.tensor([[1, self.eos_token_id]], dtype=torch.long)
        sequence_lengths = torch.tensor([2], dtype=torch.long)
        
        delta = PPOTrainer._compute_reward_shaping_delta(responses, sequence_lengths, self.eos_token_id, args)
        
        torch.testing.assert_close(delta, torch.tensor([0.8]))

    def test_default_disabled(self):
        args = PPOConfig() # All False by default
        
        responses = torch.tensor([[1, 2]], dtype=torch.long)
        sequence_lengths = torch.tensor([2], dtype=torch.long)
        
        delta = PPOTrainer._compute_reward_shaping_delta(responses, sequence_lengths, self.eos_token_id, args)
        
        torch.testing.assert_close(delta, torch.tensor([0.0]))

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            PPOConfig(reward_shaping_enable_length=True, reward_shaping_length_coef=-0.1)
        
        with self.assertRaises(ValueError):
            PPOConfig(reward_shaping_enable_repetition=True, reward_shaping_repetition_coef=-1.0)
            
        with self.assertRaises(ValueError):
            PPOConfig(reward_shaping_enable_repetition=True, reward_shaping_repetition_ngram=1)

        with self.assertRaises(ValueError):
            PPOConfig(reward_shaping_enable_eos=True, reward_shaping_eos_min_len=10, reward_shaping_eos_max_len=5)
            
if __name__ == "__main__":
    unittest.main()
