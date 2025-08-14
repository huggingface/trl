<div style="text-align: center">
<img src="https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl_banner_dark.png">
</div>

# TRL - Transformer Reinforcement Learning

TRL is a full stack library where we provide a set of tools to train transformer language models with methods like Supervised Fine-Tuning (SFT), Group Relative Policy Optimization (GRPO), Direct Preference Optimization (DPO), Reward Modeling, and more.
The library is integrated with 🤗 [transformers](https://github.com/huggingface/transformers).

## 🎉 What's New

**✨ OpenAI GPT OSS Support**: TRL now fully supports fine-tuning the latest [OpenAI GPT OSS models](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4)! Check out the:

- [OpenAI Cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)
- [GPT OSS recipes](https://github.com/huggingface/gpt-oss-recipes)
- [Our example script](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_gpt_oss.py)

You can also explore TRL-related models, datasets, and demos in the [TRL Hugging Face organization](https://huggingface.co/trl-lib).

## Learn

Learn post-training with TRL and other libraries in 🤗 [smol course](https://github.com/huggingface/smol-course).

## Contents

The documentation is organized into the following sections:

- **Getting Started**: installation and quickstart guide.
- **Conceptual Guides**: dataset formats, training FAQ, and understanding logs.
- **How-to Guides**: reducing memory usage, speeding up training, distributing training, etc.
- **Integrations**: DeepSpeed, Liger Kernel, PEFT, etc.
- **Examples**: example overview, community tutorials, etc.
- **API**: trainers, utils, etc.

## Blog posts

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <div class="aspect-[1.91/1] w-full rounded-b shadow-alternate relative flex-none overflow-hidden rounded-lg bg-white">
      <div class="absolute inset-0 group-hover:opacity-40 dark:backdrop-brightness-105"></div>
      <img src="/blog/assets/accelerate-nd-parallel/thumbnail.png" class="h-full w-full object-cover group-hover:brightness-110" alt="">
    </div>
    <div class="flex flex-col p-4">
      <h2 class="font-serif font-semibold group-hover:underline  text-xl ">Accelerate ND-Parallel: A Guide to Efficient Multi-GPU Training</h2>
      <p class="mt-3 flex items-center gap-y-1.5 font-mono text-xs text-gray-500 flex-wrap">
      <span class="inline-flex flex-none items-center">By&nbsp;<object title=""><span class="inline-block "><span class="contents"><a href="/siro1" class="hover:underline">siro1</a></span></span></object></span>
      <span class="mx-2 h-1 w-1 flex-none bg-gray-200"></span>
      <span class="truncate">August 7, 2025</span>
    </div>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/trl-vlm-alignment">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/trl_vlm/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on August 7, 2025</p>
      <span class="mx-2 h-1 w-1 flex-none bg-gray-200"></span>
      <span class="truncate">August 7, 2025</span>
      <p class="text-gray-700">Vision Language Model Alignment in TRL ⚡️</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/vllm-colocate">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/vllm-colocate/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on June 3, 2025</p>
      <p class="text-gray-700">NO GPU left behind: Unlocking Efficiency with Co-located vLLM in TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/liger-grpo">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/liger-grpo/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on May 25, 2025</p>
      <p class="text-gray-700">🐯 Liger GRPO meets TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/open-r1">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/open-r1/thumbnails.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on January 28, 2025</p>
      <p class="text-gray-700">Open-R1: a fully open reproduction of DeepSeek-R1</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/dpo_vlm">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/dpo_vlm/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on July 10, 2024</p>
      <p class="text-gray-700">Preference Optimization for Vision Language Models with TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/putting_rl_back_in_rlhf_with_rloo/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on June 12, 2024</p>
      <p class="text-gray-700">Putting RL back in RLHF</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/trl-ddpo">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/166_trl_ddpo/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on September 29, 2023</p>
      <p class="text-gray-700">Finetune Stable Diffusion Models with DDPO via TRL</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/dpo-trl">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/157_dpo_trl/dpo_thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on August 8, 2023</p>
      <p class="text-gray-700">Fine-tune Llama 2 with DPO</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/stackllama">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/138_stackllama/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on April 5, 2023</p>
      <p class="text-gray-700">StackLLaMA: A hands-on guide to train LLaMA with RLHF</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/trl-peft">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/133_trl_peft/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on March 9, 2023</p>
      <p class="text-gray-700">Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="https://huggingface.co/blog/rlhf">
      <img src="https://raw.githubusercontent.com/huggingface/blog/main/assets/120_rlhf/thumbnail.png" alt="thumbnail" class="mt-0">
      <p class="text-gray-500 text-sm">Published on December 9, 2022</p>
      <p class="text-gray-700">Illustrating Reinforcement Learning from Human Feedback</p>
    </a>
  </div>
</div>
