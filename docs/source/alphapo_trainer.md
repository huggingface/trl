# AlphaPO Trainer

## Overview

AlphaPO was introduced in [AlphaPO: Reward Shape Matters for LLM Alignment](https://arxiv.org/abs/2501.03884) by Aman Gupta, Shao Tang, et al.

The abstract from the paper is the following:

> Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Some popular examples of DAAs include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce AlphaPO, a new DAA method that leverages an α-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and overoptimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7% to 10% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B while achieving 15% to 50% relative improvement over DPO on the same models. The analysis and results presented highlight the importance of the reward shape and how one can systematically change it to affect training dynamics, as well as improve alignment performance..

It studies the crucial role of reward shape within the context of preference alignment. Altering the reward function shape to enhance the alignment performance of SimPO is a novel insight introduced and demonstrated in this paper. Since reward function (via the choice of divergence measure) occurs in other methods such as DPO and RLHF, whether it is possible that changing its shape can provide value for them too is a worthwhile future research direction

This post-training method was contributed by Qingquan Song, Shao Tang, Aman Gupta and Sirou Zhu.

## Quick start

This example demonstrates how to train a model using the AlphaPO method. We use the [Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the base model. We use the preference data from the [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback). You can view the data in the dataset here:

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

Below is the script to train the model:

```python
# train_alphapo.py
from datasets import load_dataset
from trl import AlphaPOConfig, AlphaPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = AlphaPOConfig(output_dir="Qwen2-0.5B-AlphaPO", logging_steps=10)
trainer = AlphaPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

Execute the script using the following command:

```bash
accelerate launch train_alphapo.py
```

Distributed across 8 GPUs, the training takes approximately 30 minutes. You can verify the training progress by checking the reward graph. An increasing trend in the reward margin indicates that the model is improving and generating better responses over time.

To see how the [trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-AlphaPO) performs, you can use the [Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models).

<pre><code>$ transformers-cli chat --model_name_or_path trl-lib/Qwen2-0.5B-AlphaPO
<strong><span style="color: red;">&lt;quentin_gallouedec&gt;:</span></strong>
What is the best programming language?

<strong><span style="color: blue;">&lt;trl-lib/Qwen2-0.5B-AlphaPO&gt;:</span></strong>
It's challenging to determine the best programming language as no one language is perfect, as the complexity of a task and the type of project are significant factors. Some popular languages include Java, Python, JavaScript, and
C++. If you have specific needs or requirements for a specific project, it's important to choose the language that best suits those needs.                                                                                          

Here are some other factors to consider when choosing a programming language for a project:

 <strong><span style="color: green;">• Language proficiency:</span></strong> A good programming language is more likely to be easy to understand and use, and will allow developers to collaborate on projects more efficiently.                                     
 <strong><span style="color: green;">• Ease of use:</span></strong> There are tools and libraries available to make programming more accessible, so developers should choose a language that can help them get started easier.
 <strong><span style="color: green;">• Code readability:</span></strong> A clear and concise codebase should be easy to read and understand, especially when working with large projects.
 <strong><span style="color: green;">• Tool and framework support:</span></strong> There are numerous libraries available for Python, Java, and JavaScript, along with tools like IDEs and static code analysis tools.
 <strong><span style="color: green;">• Accessibility:</span></strong> Some languages and tools have features that make them more accessible to developers with disabilities, such as support for screen readers.
 <strong><span style="color: green;">• Version control:</span></strong> As your projects grow and complexity increases, version control tools can be beneficial for tracking changes.

</code></pre>

## Expected dataset type

AlphaPO requires a [preference dataset](dataset_formats#preference). The [`AlphaPOTrainer`] supports both [conversational](dataset_formats#conversational) and [standard](dataset_formats#standard) dataset format. When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

Although the [`AlphaPOTrainer`] supports both explicit and implicit prompts, we recommend using explicit prompts. If provided with an implicit prompt dataset, the trainer will automatically extract the prompt from the `"chosen"` and `"rejected"` columns. For more information, refer to the [preference style](dataset_formats#preference) section.

## Example script

We provide an example script to train a model using the AlphaPO method. The script is available in [`examples/scripts/alphapo.py`](https://github.com/huggingface/trl/blob/main/examples/scripts/alphapo.py)

To test the AlphaPO script with the [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) on the [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized), run the following command:

```bash
accelerate launch examples/scripts/alphapo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --num_train_epochs 1 \
    --logging_steps 25 \
    --output_dir Qwen2-0.5B-AlphaPO
```

## Usage tips

### For Mixture of Experts Models: Enabling the auxiliary loss

MOEs are the most efficient if the load is about equally distributed between experts.  
To ensure that we train MOEs similarly during preference-tuning, it is beneficial to add the auxiliary loss from the load balancer to the final loss.

This option is enabled by setting `output_router_logits=True` in the model config (e.g. [`~transformers.MixtralConfig`]).  
To scale how much the auxiliary loss contributes to the total loss, use the hyperparameter `router_aux_loss_coef=...` (default: `0.001`) in the model config.

## Logged metrics

While training and evaluating we record the following reward metrics:

- `rewards/chosen`: the mean log probabilities of the policy model for the chosen responses scaled by beta
- `rewards/rejected`: the mean log probabilities of the policy model for the rejected responses scaled by beta
- `rewards/accuracies`: mean of how often the chosen rewards are > than the corresponding rejected rewards
- `rewards/margins`: the mean difference between the chosen and corresponding rejected rewards
- `log_odds_chosen`: the mean log odds ratio of the chosen responses over the rejected responses
- `log_odds_ratio`: the mean of the `log(sigmoid(log_odds_chosen))`
- `nll_loss`: the mean negative log likelihood loss from the SFT part of the loss over chosen responses
 
## AlphaPOTrainer

[[autodoc]] AlphaPOTrainer

## AlphaPOConfig

[[autodoc]] AlphaPOConfig
