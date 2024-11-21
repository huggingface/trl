# Judges

<Tip warning={true}>

TRL Judges is an experimental API which is subject to change at any time.

</Tip>

TRL provides judges to easily compare two completions.

Make sure to have installed the required dependencies by running:

```bash
pip install trl[judges]
```

## Using the provided judges

TRL provides several judges out of the box. For example, you can use the `HfPairwiseJudge` to compare two completions using a pre-trained model from the Hugging Face model hub:

```python
from trl import HfPairwiseJudge

judge = HfPairwiseJudge()
judge.judge(
    prompts=["What is the capital of France?", "What is the biggest planet in the solar system?"],
    completions=[["Paris", "Lyon"], ["Saturn", "Jupiter"]],
)  # Outputs: [0, 1]
```

## Define your own judge

To define your own judge, we provide several base classes that you can subclass. For rank-based judges, you need to subclass [`BaseRankJudge`] and implement the [`BaseRankJudge.judge`] method. For pairwise judges, you need to subclass [`BasePairJudge`] and implement the [`BasePairJudge.judge`] method. If you want to define a judge that doesn't fit into these categories, you need to subclass [`BaseJudge`] and implement the [`BaseJudge.judge`] method.

As an example, let's define a pairwise judge that prefers shorter completions:

```python
from trl import BasePairwiseJudge

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

[[autodoc]] PairRMJudge

### HfPairwiseJudge

[[autodoc]] HfPairwiseJudge

### OpenAIPairwiseJudge

[[autodoc]] OpenAIPairwiseJudge

### AllTrueJudge

[[autodoc]] AllTrueJudge

## Base classes

### BaseJudge

[[autodoc]] BaseJudge

### BaseBinaryJudge

[[autodoc]] BaseBinaryJudge

### BaseRankJudge

[[autodoc]] BaseRankJudge

### BasePairwiseJudge

[[autodoc]] BasePairwiseJudge
