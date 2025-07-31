import torch
def compute_kl(a, b, kl_penalty_type: str = "kl") -> torch.FloatTensor:
        """
        Compute KL divergence given logprob and ref_logprob.

        Args:
            logprob: Current/Old policy log probabilities
            ref_logprob: Reference policy log probabilities  
            kl_penalty_type: Type of KL penalty ("kl", "abs")
            
        Returns:
            KL divergence tensor
        """
        if kl_penalty_type in ("kl", "k1"):
            return a - b # shape: (batch_size, response_length)

        if kl_penalty_type == "abs":
            return (a - b).abs()

        raise NotImplementedError(f"KL penalty type {kl_penalty_type} not implemented")

print(compute_kl(torch.tensor([10.0, 27.0, 33.0]), torch.tensor([1.0, 2.0, 3.0]),"kl"))



def apply_kl_penalty_to_rewards(completion_mask, token_level_scores, kl, beta):
        """Apply KL penalty to the token-level rewards.

        This function computes the KL divergence between the reference policy and current policy,
        then applies a penalty to the token-level rewards based on this divergence.

        Args:
            data : Token-level reward scores
            beta: KL penalty coefficient
            kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
            
        Returns:
                token_level_rewards: The updated token-level rewards adjusted by KL penalty

        """

        kl = kl * completion_mask # (batch_size, response_length) 
        beta = beta # kl coef
        token_level_rewards = token_level_scores - beta * kl # <-- this line is including the KL penalty in the reward
        token_level_rewards_including_kl = token_level_rewards * completion_mask # (batch_size, response_length)

        return token_level_rewards_including_kl

print(apply_kl_penalty_to_rewards(
    torch.tensor([1, 1, 1]), 
    torch.tensor([10, 20, 30]),
    torch.tensor([0.1, 0.2, 0.3]),
    0.01)
    )

def convert_rewards_to_token_level(attention_mask, data, scores: torch.Tensor):
    batch_size = data.batch.batch_size[0]
    # expand as token_level_reward
    attention_mask = data["attention_mask"]
    position_ids = data["position_ids"]
    response_length = data["responses"].shape[-1]

    if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
        position_ids = position_ids[:, 0, :]

    eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
    token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
    token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

    # select the response part
    token_level_scores = token_level_scores[:, -response_length:]

    return token_level_scores



