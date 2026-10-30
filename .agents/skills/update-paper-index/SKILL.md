---
name: update-paper-index
description: Add or review a paper entry in TRL's paper index. Use when a PR implements a method, algorithm, or training approach from a research paper, or when reviewing such a PR.
---

# Update the paper index

Any PR that implements a method, algorithm, or training approach from a research paper must add a corresponding subsection to `docs/source/paper_index.md`. When reviewing such a PR, check that the file was updated.

## Entry format

The file is organized as one `##` section per method family (usually one per trainer), each holding `###` subsections, one per paper.

Each entry contains:

1. The paper title as the `###` heading.
2. A paper link line, using the Hugging Face paper page (same ID as arXiv), never an arxiv.org link:
   ```
   **📜 Paper**: https://huggingface.co/papers/<id>
   ```
3. A few sentences on what the paper introduces and how it maps to TRL.
4. A Python snippet showing the TRL config that reproduces the paper's setting, with the paper's hyperparameters quoted in comments:
   ```python
   from trl import GRPOConfig, GRPOTrainer

   training_args = GRPOConfig(
       beta=0.001,  # "the KL coefficient to 0.001"
       num_generations=16,  # "For each question, we sample 16 outputs..."
   )
   ```
   When the paper doesn't specify hyperparameters, say so in a comment rather than inventing values.

## Placement

- If the paper belongs to an existing method family, add it under that `##` section.
- If it introduces a new trainer or family, add a new `##` section with a one-line pointer to the trainer, e.g. `Papers relating to the [`GRPOTrainer`].`
