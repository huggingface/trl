# AGENTS.md

## Repository-specific guidance

### Main code vs experimental code

The repository is separated into **main code** and **experimental code**.

* **Main code** should remain stable, consistent, and well-tested.
* **Experimental code** may be less stable and may contain inconsistent patterns or limited testing.

Small non-invasive improvements that make experimental code more consistent with the main codebase are encouraged, but avoid large refactors.

### Paper implementations

If a PR implements a method, algorithm, or training approach from a research paper, it must also add a corresponding subsection to `paper_index.md`.

When reviewing such PRs, ensure that `paper_index.md` was updated.

### Code duplication and consistency

Trainers in this repository are **self-contained by design**. Shared logic (generation, reward computation, metric logging, weight syncing, etc.) is deliberately duplicated across trainers rather than abstracted into a shared base class.

This is intentional: each trainer must be readable, modifiable, and evolvable in isolation. The base class (`_BaseTrainer`) provides only minimal utilities (model card generation). Everything else — vLLM generation paths, `_get_per_token_logps_and_entropies`, `_calculate_rewards`, `_prepare_inputs`, metric logging — is copied in full.

**The tradeoff**: duplication is accepted, but **consistency is mandatory**. When the same logic appears in multiple trainers, the duplicated blocks must stay aligned:

- Same variable names (`self._last_loaded_step`, `self._metrics[mode]`, …)
- Same control flow structure (if/elif/else branches in the same order)
- Same comments (word-for-word when the logic is identical)
- Divergences only where the trainer's semantics require it (e.g., GRPO extracts logprobs from vLLM, RLOO discards them)

**Consistency over correctness**: this is a strong requirement. When duplicating code, reproduce it exactly — even if you believe the original has a bug. Do not silently fix the issue in your copy. Instead, keep your copy consistent with the source and report the problem so it can be fixed across all trainers in a dedicated PR. A correct-but-inconsistent codebase is harder to maintain than a consistently-wrong one that can be fixed in a single sweep.

**When modifying duplicated code**: if you change a pattern that exists in multiple trainers (e.g., the vLLM generation path in `_generate_single_turn`), apply the same change to all other trainers. A fix in GRPO often implies the same fix in RLOO, and vice versa. Not propagating a change is a bug.

**When reviewing**: if a PR touches duplicated logic, verify that all copies are updated consistently. A common mistake is fixing one trainer and forgetting the others.

### Simplicity

This codebase values **leanness and simplicity above all**. Prefer straightforward, inline code over abstractions, helpers, or utilities — even at the cost of some robustness or generality.

Concretely:

- Do not add layers of indirection (registries, factory patterns, plugin systems). A contributor should be able to read a trainer top to bottom and understand the full flow.
- Prefer a simple implementation that covers 90% of cases over a complex one that covers 100%. A function that handles the common path in 20 lines is better than a catch-all that handles every edge case in 80.
- Do not add defensive code, fallback paths, or configuration options "just in case". Only handle cases that actually exist today.
- Avoid `hasattr` and `getattr`. Their use is almost always a symptom of overly defensive programming or a disguised version check (e.g., "this attribute was added in version X"). Instead, either drop the conditional entirely or express the version check explicitly with a version comparison. There is nearly always a cleaner alternative.
- When in doubt, prefer less code. Every new function, parameter, or branch is maintenance burden. The best abstraction is often no abstraction.

## Documentation

### Docstrings

Docstrings must follow the repository format below. Do **not** convert docstrings to other styles (Google, NumPy, etc.).

Rules:

* Types appear in backticks inside parentheses: (`str`)
* Optional parameters are marked with `*optional*`
* Defaults are written as: `defaults to <value>`
* When the default is `None`, prefer ```(`str`, *optional*)``` instead of ```(`str` or `None`, *optional*, defaults to `None`)```
* Union types use `or`: `str` or `None`
* References to classes use the format: [`~transformers.PreTrainedModel`]
* Class docstrings may group parameters using headers such as: `> Parameters for X:`

Example:

````python
def method(self, param1: str, param2: int = 1, param3: float | None = None):
    """
    Brief one-line description of what this does.

    Args:
        param1 (`str`):
            Description of required param.
        param2 (`int`, *optional*, defaults to `1`):
            Description of optional param with default.
        param3 (`float`, *optional*):
            Description of optional param without explicit default.

    Returns:
        `dict` with keys:
            - `key1` (`list[int]`):
                Description of this key.

    Examples:

    ```python
    >>> my_func("hello")
    ```
    """
````

### Links to papers

When linking to papers, use `https://huggingface.co/papers/<id>` instead of `https://arxiv.org/abs/<id>` (same ID suffix system).
