
GPT OSS is a hugely anticipated open-weights release by OpenAI, designed for powerful reasoning, agentic tasks, and versatile developer use cases. It comprises two models: a big one with 117B parameters (gpt-oss-120b), and a smaller one with 21B parameters (gpt-oss-20b). Both are mixture-of-experts (MoEs) and use a 4-bit quantization scheme (MXFP4), enabling fast inference (thanks to fewer active parameters, see details below) while keeping resource usage low. The large model fits on a single H100 GPU, while the small one runs within 16GB of memory and is perfect for consumer hardware and on-device applications.


To make it even better and more impactful for the community, the models are licensed under the Apache 2.0 license, along with a minimal usage policy:
>  We aim for our tools to be used safely, responsibly, and democratically, while maximizing your control over how you use them. By using gpt-oss, you agree to comply with all applicable law.

According to OpenAI, this release is a meaningful step in their commitment to the open-source ecosystem, in line with their stated mission to make the benefits of AI broadly accessible. Many use cases rely on private and/or local deployments, and we at Hugging Face are super excited to welcome OpenAI to the community. We believe these will be long-lived, inspiring and impactful models.


# Fine-Tuning OpenAI GPT OSS with TRL

## Introduction

OpenAI is releasing its first major [open-source models](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) in years!
Dans ce blog post, on va rapidement présenter ces deux modèles ([`gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) et [`gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)), le nouveau format de chat introduit conjointement nommé Harmony, et présenterons un exemple de fine-tuning sur une tâche simple de raisonnement cross-lingual. C'est parti!

## Understanding the Model

La release contient deux modèles, un de 117B paramètres ([`gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b)) et un de 21B parametees ([`gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b)), qui sont des modèles de type "Mixture of Experts" (MoE). Cela signifie qu'ils utilisent une architecture où seuls certains _experts_ sont activés pour chaque requête. 

What’s happening: OpenAI is releasing its first major open-source model in years.
Why it matters: It’s instruction-tuned, open, and powerful—but needs fine-tuning for specific tasks.
What we’ll do: First introduce the model and its new chat format "harmony", then show how to fine-tune it using TRL on a multilingual reasoning task.

High-level overview:
It’s an Instruct model.
It uses a Mixture of Experts (MoE) architecture.
...
## 3. What Is the Harmony Format?
Quick explanation:
A new prompt/response format OpenAI uses internally for better instruction-following.
You must format your data using Harmony to get good performance.
Explain the format:
thinking, developper, ...
Examples:
Include 2–3 formatted examples, including multilingual examples.
## 4. Designing the Task: Cross-Lingual Reasoning
Task definition:
The model is asked a question in Language A and must reason in Language B, then answer in Language A.
Motivation:
Improve interpretability
Easy task
Most base models struggle with this unless explicitly trained on it.
Data structure:
Show how to build a dataset in this format.
Include 1–2 real examples.
## 5. Running the Base Model
Load and generate:
Load the model.
Run a few examples on your task without fine-tuning.
Show limitations:
Model might ignore the target language or answer incorrectly.
Use this to motivate fine-tuning.
## 6. Fine-Tuning with TRL
Installation instructions
Code snippet:
Show a minimal example.
Explain key parts: tokenizer, formatting function, training args.
Launch command:
Include CLI command to run the script locally or on a cluster.
8 H100 and/or 1 H100?
Includetrackio embed curves
## 7. Results & Generalization
Show qualitative outputs:
Pre- vs post-finetune examples.
Include languages the model never saw during training to show generalization.
HF demo space + link to trained model
## 8. Conclusion