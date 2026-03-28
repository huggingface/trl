# Per-Sample Tool Filtering in GRPOTrainer

## Motivation

In many agentic settings, different training samples require different subsets of tools.
For example, a math question might only need a `calculator`, while a translation task
might only need a `translator`. Exposing all tools on every sample can confuse the model
and dilute the training signal.

`GRPOTrainer` now supports an optional `tools_column_name` parameter that lets you specify
**which tools are available per dataset sample**, drawn from the global `tools` pool.

## How It Works

1. **Global tool pool** — You pass the full set of tools to the trainer via `tools=[...]` as before.
2. **Per-sample tool column** — Your dataset includes a column (e.g., `"tools"`) containing a
   list of tool **names** (strings matching `tool.__name__`) allowed for each sample.
3. **Automatic filtering** — For each rollout, only the specified tools appear in the model's
   system prompt (chat template) and are available for execution. If the column is missing or
   `None` for a sample, all tools are used as a fallback.

## Example

```python
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig


# Define tool functions
def calculator(number_a: float, operation: str, number_b: float) -> str:
    """Perform a basic arithmetic operation on two numbers.

    Args:
        number_a: The first operand.
        operation: The operation to perform. One of '+', '-', '*', '/'.
        number_b: The second operand.

    Returns:
        The result of the operation as a string.

    Raises:
        ValueError: If the operation is not supported or division by zero is attempted.
    """
    try:
        number_a = float(number_a)
    except (TypeError, ValueError):
        raise TypeError(f"number_a must be convertible to a number, got {type(number_a).__name__!r}")
    try:
        number_b = float(number_b)
    except (TypeError, ValueError):
        raise TypeError(f"number_b must be convertible to a number, got {type(number_b).__name__!r}")
    if operation == "+":
        return str(number_a + number_b)
    elif operation == "-":
        return str(number_a - number_b)
    elif operation == "*":
        return str(number_a * number_b)
    elif operation == "/":
        if number_b == 0:
            raise ValueError("Division by zero is not allowed.")
        return str(number_a / number_b)
    else:
        raise ValueError(f"Unsupported operation '{operation}'. Use one of: +, -, *, /")


def translator(text: str, target_language: str) -> str:
    """Translate text to a target language.

    Args:
        text: The text to translate.
        target_language: ISO language code, e.g. 'fr', 'es', 'de'.

    Returns:
        The translated text.
    """
    # Placeholder — in practice, call a translation API
    return f"[{target_language}] {text}"


# Build dataset with per-sample tool column
dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What is 123 * 456?"}],
        [{"role": "user", "content": "Translate 'good morning' to French."}],
        [{"role": "user", "content": "Compute 2^10 and translate the result to Spanish."}],
    ],
    "tools": [
        ["calculator"],            # only calculator available
        ["translator"],            # only translator available
        ["calculator", "translator"],  # both available
    ],
})

# Create trainer with per-sample tool filtering
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=my_reward,
    tools=[calculator, translator],
    tools_column_name="tools",       # <-- activates per-sample filtering
    train_dataset=dataset,
)

trainer.train()
```

## Validation

At initialization, the trainer validates that every tool name referenced in the dataset column
exists in the global `tools` pool. If an unknown tool name is found, a clear `ValueError` is
raised listing the offending names and the available pool.

## Backward Compatibility

When `tools_column_name=None` (the default), behavior is identical to the existing API — all
tools in the `tools` list are used for every sample.
