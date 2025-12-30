<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_dark.png">
</div>

# TRL - Transformer Reinforcement Learning

TRL is a full stack library for post-training and aligning transformer language models. From Supervised Fine-Tuning (SFT) to Reinforcement Learning (GRPO, PPO) and Preference Optimization (DPO, KTO), TRL provides everything you need.

<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.5rem 0;">
  <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.25rem 0.75rem; border-radius: 2rem; font-size: 0.85rem;">🚀 vLLM integration</span>
  <span style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 0.25rem 0.75rem; border-radius: 2rem; font-size: 0.85rem;">🔧 PEFT/LoRA ready</span>
  <span style="background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); color: white; padding: 0.25rem 0.75rem; border-radius: 2rem; font-size: 0.85rem;">👁️ VLM support</span>
  <span style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 0.25rem 0.75rem; border-radius: 2rem; font-size: 0.85rem;">📊 Trackio & WandB</span>
  <span style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 0.25rem 0.75rem; border-radius: 2rem; font-size: 0.85rem;">🤗 Transformers native</span>
</div>

## Quick Start

<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin: 1rem 0;">
  <a href="installation" style="text-decoration: none; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 0.5rem 1rem; border-radius: 0.5rem; color: #374151; font-weight: 500; text-align: center;">📦 Installation</a>
  <a href="quickstart" style="text-decoration: none; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 0.5rem 1rem; border-radius: 0.5rem; color: #374151; font-weight: 500; text-align: center;">⚡ Quickstart</a>
  <a href="dataset_formats" style="text-decoration: none; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 0.5rem 1rem; border-radius: 0.5rem; color: #374151; font-weight: 500; text-align: center;">📚 Conceptual Guides</a>
  <a href="clis" style="text-decoration: none; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 0.5rem 1rem; border-radius: 0.5rem; color: #374151; font-weight: 500; text-align: center;">🛠️ How-to Guides</a>
  <a href="deepspeed_integration" style="text-decoration: none; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 0.5rem 1rem; border-radius: 0.5rem; color: #374151; font-weight: 500; text-align: center;">🔌 Integrations</a>
  <a href="example_overview" style="text-decoration: none; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 0.5rem 1rem; border-radius: 0.5rem; color: #374151; font-weight: 500; text-align: center;">💡 Examples</a>
  <a href="sft_trainer" style="text-decoration: none; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 0.5rem 1rem; border-radius: 0.5rem; color: #374151; font-weight: 500; text-align: center;">📖 API Reference</a>
  <a href="experimental_overview" style="text-decoration: none; background: #f3f4f6; border: 1px solid #e5e7eb; padding: 0.5rem 1rem; border-radius: 0.5rem; color: #374151; font-weight: 500; text-align: center;">🧪 Experimental</a>
</div>

## Latest News

<div class="mt-10">
  <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.75rem;">
    <a class="!no-underline border dark:border-gray-700 rounded-lg shadow hover:shadow-lg" href="https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_trl_lora_qlora.ipynb" style="display: block; padding: 0.4rem 0.75rem; margin: 1%;">
      <img src="https://github.com/huggingface/trl/blob/new-docs-landing/assets/1.png?raw=true" alt="thumbnail" style="width: 100%; height: auto; border-radius: 0.25rem; display: block;">
      <p class="text-gray-700" style="margin: 0.4rem 0 0 0;">GRPO using QLoRA on free Colab</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 rounded-lg shadow hover:shadow-lg" href="https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_functiongemma_browsergym_openenv.ipynb" style="display: block; padding: 0.4rem 0.75rem; margin: 1%;">
      <img src="https://github.com/huggingface/trl/blob/new-docs-landing/assets/2.png?raw=true" alt="thumbnail" style="width: 100%; height: auto; border-radius: 0.25rem; display: block;">
      <p class="text-gray-700" style="margin: 0.4rem 0 0 0;">GRPO FunctionGemma in BrowserGym (OpenEnv)</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 rounded-lg shadow hover:shadow-lg" href="https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_rnj_1_instruct.ipynb" style="display: block; padding: 0.4rem 0.75rem; margin: 1%;">
      <img src="https://github.com/huggingface/trl/blob/new-docs-landing/assets/3.png?raw=true" alt="thumbnail" style="width: 100%; height: auto; border-radius: 0.25rem; display: block;">
      <p class="text-gray-700" style="margin: 0.4rem 0 0 0;">GRPO rnj-1-instruct with QLoRA for reasoning</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 rounded-lg shadow hover:shadow-lg" href="https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_ministral3_vl.ipynb" style="display: block; padding: 0.4rem 0.75rem; margin: 1%;">
      <img src="https://github.com/huggingface/trl/blob/new-docs-landing/assets/4.png?raw=true" alt="thumbnail" style="width: 100%; height: auto; border-radius: 0.25rem; display: block;">
      <p class="text-gray-700" style="margin: 0.4rem 0 0 0;">SFT Ministral-3 VL with QLoRA</p>
    </a>
  </div>
</div>

## Taxonomy

Available TRL trainers organized by method type (⚡️ = vLLM support; 🧪 = experimental).

<div style="display: flex; justify-content: space-between; width: 100%; gap: 2rem;">
<div style="flex: 1; min-width: 0;">

### Online methods

- [`GRPOTrainer`](grpo_trainer) ⚡️
- [`RLOOTrainer`](rloo_trainer) ⚡️
- [`OnlineDPOTrainer`](online_dpo_trainer) 🧪 ⚡️
- [`NashMDTrainer`](nash_md_trainer) 🧪 ⚡️
- [`PPOTrainer`](ppo_trainer) 🧪
- [`XPOTrainer`](xpo_trainer) 🧪 ⚡️

### Reward modeling

- [`RewardTrainer`](reward_trainer)
- [`PRMTrainer`](prm_trainer) 🧪

</div>
<div style="flex: 1; min-width: 0;">

### Offline methods

- [`SFTTrainer`](sft_trainer)
- [`DPOTrainer`](dpo_trainer)
- [`BCOTrainer`](bco_trainer) 🧪
- [`CPOTrainer`](cpo_trainer) 🧪
- [`KTOTrainer`](kto_trainer) 🧪
- [`ORPOTrainer`](orpo_trainer) 🧪

### Knowledge distillation

- [`GKDTrainer`](gkd_trainer) 🧪
- [`MiniLLMTrainer`](minillm_trainer) 🧪

</div>
</div>

Explore TRL models, datasets, and demos in the [TRL Hugging Face organization](https://huggingface.co/trl-lib).

## Blog Posts

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/openenv">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/openenv/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">October 23, 2025</p>
      <p class="text-gray-700">Building the Open Agent Ecosystem Together: Introducing OpenEnv</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/trl-vlm-alignment">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/trl_vlm/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">August 7, 2025</p>
      <p class="text-gray-700">Vision Language Model Alignment in TRL ⚡️</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/vllm-colocate">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/vllm-colocate/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">June 3, 2025</p>
      <p class="text-gray-700">NO GPU left behind: Unlocking Efficiency with Co-located vLLM in TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/liger-grpo">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/liger-grpo/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">May 25, 2025</p>
      <p class="text-gray-700">🐯 Liger GRPO meets TRL</p>
    </a>
  </div>
</div>

<details>
<summary style="cursor: pointer; font-weight: 500; color: #6b7280; margin-top: 1rem;">Show older posts...</summary>

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/open-r1">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/open-r1/thumbnails.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">January 28, 2025</p>
      <p class="text-gray-700">Open-R1: a fully open reproduction of DeepSeek-R1</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/dpo_vlm">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/dpo_vlm/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">July 10, 2024</p>
      <p class="text-gray-700">Preference Optimization for Vision Language Models with TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/putting_rl_back_in_rlhf_with_rloo/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">June 12, 2024</p>
      <p class="text-gray-700">Putting RL back in RLHF</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/trl-ddpo">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/166_trl_ddpo/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">September 29, 2023</p>
      <p class="text-gray-700">Finetune Stable Diffusion Models with DDPO via TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/dpo-trl">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/157_dpo_trl/dpo_thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">August 8, 2023</p>
      <p class="text-gray-700">Fine-tune Llama 2 with DPO</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/stackllama">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/138_stackllama/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">April 5, 2023</p>
      <p class="text-gray-700">StackLLaMA: A hands-on guide to train LLaMA with RLHF</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/trl-peft">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/133_trl_peft/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">March 9, 2023</p>
      <p class="text-gray-700">Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/rlhf">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/120_rlhf/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">December 9, 2022</p>
      <p class="text-gray-700">Illustrating Reinforcement Learning from Human Feedback</p>
    </a>
  </div>
</div>

</details>

## Talks

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/Fine%20tuning%20with%20TRL%20(Oct%2025).pdf">
      <img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/Fine%20tuning%20with%20TRL%20(Oct%2025).png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">October 30, 2025</p>
      <p class="text-gray-700">Fine tuning with TRL</p>
    </a>
  </div>
</div>
