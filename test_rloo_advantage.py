import torch

def test_rloo_advantage_calculation():
    """Test that RLOO advantage calculation matches the expected leave-one-out baseline."""
    
    # Example rewards: 2 prompts, 3 generations each
    # Prompt 1: [1.0, 2.0, 3.0] -> baselines: [2.5, 2.0, 1.5]
    # Prompt 2: [4.0, 5.0, 6.0] -> baselines: [5.5, 5.0, 4.5]
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    num_generations = 3
    
    # RLOO advantage calculation
    rewards_reshaped = rewards.view(-1, num_generations)  # Shape: (2, 3)
    print(f"rewards_reshaped: {rewards_reshaped}")
    total_rewards = rewards_reshaped.sum(dim=1, keepdim=True)  # Shape: (2, 1)
    print(f"total_rewards: {total_rewards}")
    a = (total_rewards - rewards_reshaped)
    print(f"a: {a}")
    baseline = a / (num_generations - 1)  # Shape: (2, 3)
    print(f"baseline: {baseline}")
    advantages_reshaped = rewards_reshaped - baseline  # Shape: (2, 3)
    print(f"advantages_reshaped: {advantages_reshaped} = {rewards_reshaped} - {baseline}")
    advantages = advantages_reshaped.flatten()
    
    # Expected calculations:
    # Prompt 1: rewards [1.0, 2.0, 3.0], total = 6.0
    # baseline[0] = (6.0 - 1.0) / 2 = 2.5, advantage[0] = 1.0 - 2.5 = -1.5
    # baseline[1] = (6.0 - 2.0) / 2 = 2.0, advantage[1] = 2.0 - 2.0 = 0.0  
    # baseline[2] = (6.0 - 3.0) / 2 = 1.5, advantage[2] = 3.0 - 1.5 = 1.5
    
    # Prompt 2: rewards [4.0, 5.0, 6.0], total = 15.0
    # baseline[0] = (15.0 - 4.0) / 2 = 5.5, advantage[0] = 4.0 - 5.5 = -1.5
    # baseline[1] = (15.0 - 5.0) / 2 = 5.0, advantage[1] = 5.0 - 5.0 = 0.0
    # baseline[2] = (15.0 - 6.0) / 2 = 4.5, advantage[2] = 6.0 - 4.5 = 1.5
    
    expected_advantages = torch.tensor([-1.5, 0.0, 1.5, -1.5, 0.0, 1.5])
    
    print("Rewards:", rewards)
    print("Calculated advantages:", advantages)
    print("Expected advantages:", expected_advantages)
    print("Match:", torch.allclose(advantages, expected_advantages, atol=1e-6))

if __name__ == "__main__":
    test_rloo_advantage_calculation() 