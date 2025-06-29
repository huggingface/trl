import torch

def compare_grpo_vs_rloo_baselines():
    
    # Example: 2 prompts, 3 generations each
    num_generations = 3
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) #[p0_com0, p0_com1, p0_com2, p1_com0, p1_com1, p1_com2]
    print(f"rewards: {rewards}")

    
    # GRPO approach: group-wise baseline
    rewards_reshaped = rewards.view(-1, num_generations)  # (2, 3)
    print(f"Rewards reshaped: {rewards_reshaped}")
    
    # GRPO: compute mean per group (prompt)
    mean_grouped_rewards = rewards_reshaped.mean(dim=1)  # (2,) 

    # GRPO uses repeat_interleave to broadcast gp means
    grpo_baseline = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)  # (6,)
    
    # GRPO adv
    grpo_advantages = rewards - grpo_baseline
    print(f"GRPO adv: {grpo_advantages}")

    # RLOO: compute leave-one-out baseline for each individual sample
    total_rewards = rewards_reshaped.sum(dim=1, keepdim=True)  # (2, 1)
    print(f"Total rewards per prompt: {total_rewards.squeeze()}")
    
    rloo_baseline = (total_rewards - rewards_reshaped) / (num_generations - 1)  # (2, 3)
    print(f"RLOO baseline (reshaped): {rloo_baseline}")
    
    # No repeat_interleave needed - we already have the right shape!
    rloo_baseline = rloo_baseline.flatten()  # (6,)
    print(f"RLOO baseline (flattened): {rloo_baseline}")
    
    rloo_advantages = rewards - rloo_baseline
    print(f"RLOO advantages: {rloo_advantages}")
    print()
    
    # Key difference explanation
    print("Key Differences:")
    print("-" * 30)
    print("GRPO: All samples in a group share the SAME baseline (group mean)")
    print(f"  - Prompt 0 samples all use baseline: {mean_grouped_rewards[0]:.4f}")
    print(f"  - Prompt 1 samples all use baseline: {mean_grouped_rewards[1]:.4f}")
    print()
    print("RLOO: Each sample has a DIFFERENT baseline (leave-one-out mean)")
    print(f"  - Prompt 0, Gen 0: baseline = {rloo_baseline[0]:.4f}")
    print(f"  - Prompt 0, Gen 1: baseline = {rloo_baseline[1]:.4f}")  
    print(f"  - Prompt 0, Gen 2: baseline = {rloo_baseline[2]:.4f}")
    print(f"  - Prompt 1, Gen 0: baseline = {rloo_baseline[3]:.4f}")
    print(f"  - Prompt 1, Gen 1: baseline = {rloo_baseline[4]:.4f}")
    print(f"  - Prompt 1, Gen 2: baseline = {rloo_baseline[5]:.4f}")
    print()
    
if __name__ == "__main__":
    compare_grpo_vs_rloo_baselines()