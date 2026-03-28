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
import ast
import operator
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig


# Define tool functions
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression to evaluate, e.g. '2 + 2'.
        Supports basic arithmetic with +, -, *, /, and parentheses.

    Returns:
        The result of the expression as a string.

    Raises:
        ValueError: If the expression contains unsupported syntax.
    """
    _allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _allowed_operators:
            return _allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _eval(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Unsupported expression for calculator tool.")

    return str(_eval(ast.parse(expression, mode="eval")))


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
