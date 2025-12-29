# HICRA Trainer

[![model badge](https://img.shields.io/badge/All_models-HICRA-blue)](https://huggingface.co/models?other=hicra,trl)

## Overview

TRL supports the HICRA (Hierarchy-Aware Credit Assignment) Trainer for training language models with hierarchical reasoning capabilities. HICRA is described in the paper [Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](https://huggingface.co/papers/2509.03646) by researchers at TIGER-AI-Lab.

The abstract from the paper is the following:

> We introduce HICRA (Hierarchy-Aware Credit Assignment), a novel reinforcement learning approach that enables Large Language Models (LLMs) to develop emergent hierarchical reasoning capabilities. By focusing optimization pressure on high-level strategic planning tokens (Strategic Grams), HICRA accelerates the development of advanced reasoning patterns more efficiently than standard credit assignment methods. Our experiments demonstrate that HICRA-trained models achieve superior performance on mathematical reasoning tasks while exhibiting clear separation between planning and execution phases.

HICRA extends the GRPO (Group Relative Policy Optimization) algorithm by amplifying the learning signal for strategic planning tokens, enabling models to develop hierarchical reasoning capabilities where high-level planning guides low-level execution.

This implementation was contributed by the TRL team and is based on the [VeRL implementation](https://github.com/TIGER-AI-Lab/Hierarchical-Reasoner) from TIGER-AI-Lab.

## Quick start

This example demonstrates how to train a model using the HICRA method. We train a [Qwen 0.5B Instruct model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) with the prompts from the [DeepMath-103K dataset](https://huggingface.co/datasets/trl-lib/DeepMath-103K).

```python
# train_hicra.py
from datasets import load_dataset
from trl import HICRATrainer, HICRAConfig
from trl.rewards import accuracy_reward

dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

config = HICRAConfig(
    hicra_alpha=0.2,  # Amplification factor for planning tokens
    use_hicra=True,   # Enable HICRA advantage modification
)

trainer = HICRATrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=config,
    train_dataset=dataset,
)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_hicra.py
```

## Understanding HICRA

HICRA builds on GRPO by introducing **hierarchy-aware credit assignment** that focuses optimization on strategic planning tokens. The key insight is that RL training exhibits a two-phase dynamic:

1. **Phase 1**: Consolidation of low-level procedural skills (execution tokens)
2. **Phase 2**: Exploration and mastery of high-level strategic planning (planning tokens)

By amplifying the learning signal for planning tokens, HICRA accelerates Phase 2 and unlocks advanced reasoning more efficiently.

### Key Concepts

#### Strategic Grams (SGs)

Strategic Grams are n-grams that function as high-level strategic moves in reasoning. Examples include:
- "let's try a different approach"
- "notice that"
- "we can use the fact that"
- "the key insight is"

These phrases guide the logical flow of reasoning and represent planning decisions rather than execution steps.

#### Planning vs Execution Tokens

- **Planning tokens**: Tokens that are part of Strategic Grams and guide high-level reasoning strategy
- **Execution tokens**: Tokens that perform low-level procedural steps (calculations, substitutions, etc.)

HICRA identifies planning tokens and amplifies their advantages to focus learning on strategic reasoning.

### The HICRA Algorithm

HICRA modifies the advantage computation in GRPO by applying amplification to planning tokens:

#### Paper's Algorithm

The paper defines the HICRA advantage as:

$\hat{A}_{\text{HICRA}_{i,t}} = \hat{A}_{i,t} + \alpha \cdot |\hat{A}_{i,t}|$ if $t \in S_i$ (planning tokens)

$\hat{A}_{\text{HICRA}_{i,t}} = \hat{A}_{i,t}$ if $t \notin S_i$ (execution tokens)

Where:
- $\hat{A}_{i,t}$ is the GRPO advantage
- $S_i$ is the set of planning token indices
- $\alpha \in (0, 1)$ is the amplification factor (paper uses $\alpha=0.2$)

#### VeRL Implementation

Our implementation follows the VeRL version, which extends the paper's algorithm with additional heuristics:

1. **Entropy-based token selection**: Amplifies high-entropy tokens (top-k percentile) as a proxy for planning tokens
2. **Length-based filtering**: Only amplifies advantages for responses that are:
   - Correct (advantage > 0)
   - Longer than the average length in the GRPO group
3. **Signed amplification**: Uses $\hat{A} \times (1 + \alpha \times \text{sign}(\hat{A}))$ for more nuanced credit assignment

This implementation represents the actual working code that produced the paper's results.

### HICRA Training Flow

1. **Generate completions**: Sample prompts and generate multiple completions per prompt (same as GRPO)
2. **Compute advantages**: Calculate GRPO advantages based on rewards
3. **Identify planning tokens**: 
   - Use entropy threshold to identify high-entropy tokens
   - Optionally use Strategic Grams for explicit planning token identification
4. **Amplify advantages**: Apply HICRA amplification to planning/high-entropy tokens
5. **Update policy**: Use modified advantages for policy gradient computation

## Configuration

HICRA extends [`GRPOConfig`] with additional parameters:

### HICRA-Specific Parameters

- `hicra_alpha` (`float`, defaults to `0.2`): Amplification factor for planning/high-entropy tokens. Higher values increase the focus on strategic reasoning.
- `use_hicra` (`bool`, defaults to `True`): Enable/disable HICRA advantage modification. When `False`, behaves exactly like GRPO.
- `hicra_entropy_topk` (`float`, defaults to `0.3`): Top-k percentile for entropy threshold computation. Tokens with entropy above this threshold are considered high-entropy.
- `use_planning_tokens` (`bool`, defaults to `False`): Whether to use Strategic Gram-based planning token identification in addition to entropy-based selection.

### Strategic Gram Configuration

- `strategic_grams_path` (`str`, optional): Path to a JSON file containing pre-computed Strategic Grams.
- `strategic_grams` (`list[str]`, optional): Direct list of Strategic Grams to use.
- `sg_n_range` (`tuple[int, int]`, defaults to `(3, 5)`): N-gram range for Strategic Gram extraction.

### Logging Configuration

- `log_semantic_entropy` (`bool`, defaults to `True`): Log semantic entropy metrics measuring strategic diversity.
- `log_planning_token_ratio` (`bool`, defaults to `True`): Log the percentage of tokens identified as planning tokens.

## Example: Using Strategic Grams

To use Strategic Grams for explicit planning token identification:

```python
from trl import HICRATrainer, HICRAConfig
from trl.rewards import accuracy_reward

# Option 1: Use default Strategic Grams for math domain
config = HICRAConfig(
    use_planning_tokens=True,  # Enable Strategic Gram-based identification
    hicra_alpha=0.2,
)

# Option 2: Provide custom Strategic Grams
config = HICRAConfig(
    use_planning_tokens=True,
    strategic_grams=[
        "let's try a different approach",
        "notice that",
        "we can use the fact that",
        "the key insight is",
    ],
    hicra_alpha=0.2,
)

# Option 3: Load from file
config = HICRAConfig(
    use_planning_tokens=True,
    strategic_grams_path="path/to/strategic_grams.json",
    hicra_alpha=0.2,
)

trainer = HICRATrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=config,
    train_dataset=dataset,
)
```

## Extracting Strategic Grams

You can extract Strategic Grams from your own corpus using the provided utility script:

```bash
python examples/scripts/extract_strategic_grams.py \
    --dataset_name trl-lib/DeepMath-103K \
    --output_path strategic_grams_math.json \
    --n_min 3 \
    --n_max 5 \
    --n_clusters 100 \
    --top_percentile 0.2
```

The extraction process uses semantic clustering:
1. Extract n-grams (n ∈ [3, 5]) from the corpus
2. Embed n-grams using sentence transformers
3. Cluster embeddings using KMeans
4. Compute Cluster Document Frequency (CDF) for each cluster
5. Select clusters in the top 20% of CDF
6. Return n-grams from selected clusters as Strategic Grams

For more details, see the [Strategic Gram extraction example](https://github.com/huggingface/trl/blob/main/examples/scripts/extract_strategic_grams.py).

## Logged Metrics

In addition to all GRPO metrics, HICRA logs the following:

- `hicra/planning_token_ratio`: Percentage of tokens identified as planning tokens
- `hicra/planning_advantage_mean`: Average advantage for planning tokens
- `hicra/execution_advantage_mean`: Average advantage for execution tokens
- `hicra/semantic_entropy`: Shannon entropy over Strategic Gram frequency distribution (if `log_semantic_entropy=True`)

These metrics help you monitor:
- How much of the model's reasoning is strategic vs procedural
- Whether planning tokens receive stronger learning signals
- The diversity of strategic patterns used by the model

## Comparison with GRPO

HICRA is a direct extension of GRPO. To compare HICRA with standard GRPO:

```python
# GRPO baseline
grpo_config = HICRAConfig(use_hicra=False)
grpo_trainer = HICRATrainer(model=model, args=grpo_config, ...)

# HICRA
hicra_config = HICRAConfig(use_hicra=True, hicra_alpha=0.2)
hicra_trainer = HICRATrainer(model=model, args=hicra_config, ...)
```

Expected benefits of HICRA over GRPO:
- Faster development of hierarchical reasoning capabilities
- Better separation between planning and execution phases
- More efficient exploration of strategic reasoning patterns
- Improved performance on complex reasoning tasks

## Advanced Usage

### Adjusting the Amplification Factor

The `hicra_alpha` parameter controls how much to amplify planning token advantages:

```python
# Conservative amplification (closer to GRPO)
config = HICRAConfig(hicra_alpha=0.1)

# Standard amplification (paper default)
config = HICRAConfig(hicra_alpha=0.2)

# Aggressive amplification (stronger focus on planning)
config = HICRAConfig(hicra_alpha=0.3)
```

### Entropy Threshold Tuning

The `hicra_entropy_topk` parameter controls which tokens are considered high-entropy:

```python
# More selective (only top 20% entropy tokens)
config = HICRAConfig(hicra_entropy_topk=0.2)

# Standard (top 30% entropy tokens)
config = HICRAConfig(hicra_entropy_topk=0.3)

# More inclusive (top 40% entropy tokens)
config = HICRAConfig(hicra_entropy_topk=0.4)
```

### Combining with vLLM

HICRA supports all GRPO features, including vLLM-powered generation:

```python
config = HICRAConfig(
    use_hicra=True,
    hicra_alpha=0.2,
    use_vllm=True,
    vllm_mode="colocate",
)

trainer = HICRATrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=accuracy_reward,
    args=config,
    train_dataset=dataset,
)
```

For more details on vLLM integration, see the [GRPO documentation](grpo_trainer#speed-up-training-with-vllm-powered-generation).

## Implementation Details

### Relationship to Paper Algorithm

Our implementation follows the VeRL version rather than the paper's algorithm directly. Key differences:

1. **Entropy-based selection**: VeRL uses token entropy as a proxy for planning tokens, not just Strategic Grams
2. **Length filtering**: VeRL only amplifies longer-than-average correct responses
3. **Signed amplification**: VeRL uses signed amplification for more nuanced credit assignment

These modifications represent practical improvements discovered during implementation and are the actual code that produced the paper's results.

### Memory and Compute Overhead

HICRA adds minimal overhead compared to GRPO:
- Planning token identification: O(batch_size × seq_len × num_strategic_grams)
- Advantage modification: O(batch_size × seq_len)
- Additional memory: Negligible (only stores planning token masks)

The overhead is typically <5% of total training time.

## API Documentation

### HICRAConfig

[[autodoc]] HICRAConfig

### HICRATrainer

[[autodoc]] HICRATrainer

## References

- Paper: [Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](https://huggingface.co/papers/2509.03646)
- VeRL Implementation: [TIGER-AI-Lab/Hierarchical-Reasoner](https://github.com/TIGER-AI-Lab/Hierarchical-Reasoner)
- Base Algorithm: [GRPO Trainer](grpo_trainer)

## Citation

```bibtex
@article{hicra2025,
  title={Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning},
  author={TIGER-AI-Lab},
  journal={arXiv preprint arXiv:2509.03646},
  year={2025}
}
```
