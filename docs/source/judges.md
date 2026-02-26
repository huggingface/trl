# Judges

> [!WARNING]
> TRL Judges is an experimental API which is subject to change at any time. As of TRL v1.0, judges have been moved to the `trl.experimental.judges` module.

TRL provides judges to easily compare two completions.

Make sure to have installed the required dependencies by running:

```bash
pip install trl[judges]
```

## Using the provided judges

TRL provides several judges out of the box. For example, you can use the [`experimental.judges.HfPairwiseJudge`] to compare two completions using a pre-trained model from the Hugging Face model hub:

```python
from trl.experimental.judges import HfPairwiseJudge

judge = HfPairwiseJudge()
judge.judge(
    prompts=["What is the capital of France?", "What is the biggest planet in the solar system?"],
    completions=[["Paris", "Lyon"], ["Saturn", "Jupiter"]],
)  # Outputs: [0, 1]
```

## Define your own judge

To define your own judge, we provide several base classes that you can subclass. For rank-based judges, you need to subclass [`experimental.judges.BaseRankJudge`] and implement the [`experimental.judges.BaseRankJudge.judge`] method. For pairwise judges, you need to subclass [`experimental.judges.BasePairJudge`] and implement the [`experimental.judges.BasePairJudge.judge`] method. If you want to define a judge that doesn't fit into these categories, you need to subclass [`experimental.judges.BaseJudge`] and implement the [`experimental.judges.BaseJudge.judge`] method.

As an example, let's define a pairwise judge that prefers shorter completions:

```python
from trl.experimental.judges import BasePairwiseJudge

class PrefersShorterJudge(BasePairwiseJudge):
    def judge(self, prompts, completions, shuffle_order=False):
        return [0 if len(completion[0]) > len(completion[1]) else 1 for completion in completions]
```

You can then use this judge as follows:

```python
judge = PrefersShorterJudge()
judge.judge(
    prompts=["What is the capital of France?", "What is the biggest planet in the solar system?"],
    completions=[["Paris", "The capital of France is Paris."], ["Jupiter is the biggest planet in the solar system.", "Jupiter"]],
)  # Outputs: [0, 1]
```

## Provided judges

### PairRMJudge

[[autodoc]] experimental.judges.PairRMJudge

### HfPairwiseJudge

[[autodoc]] experimental.judges.HfPairwiseJudge

### OpenAIPairwiseJudge

[[autodoc]] experimental.judges.OpenAIPairwiseJudge

### AllTrueJudge

[[autodoc]] experimental.judges.AllTrueJudge

## Base classes

### BaseJudge

[[autodoc]] experimental.judges.BaseJudge

### BaseBinaryJudge

[[autodoc]] experimental.judges.BaseBinaryJudge

### BaseRankJudge

[[autodoc]] experimental.judges.BaseRankJudge

### BasePairwiseJudge

[[autodoc]] experimental.judges.BasePairwiseJudge
