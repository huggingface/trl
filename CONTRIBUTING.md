# How to contribute to TRL?

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping
others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
[code of conduct](https://github.com/huggingface/trl/blob/main/CODE_OF_CONDUCT.md).

**This guide was heavily inspired by the awesome [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to TRL:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Implement trainers for new post-training algorithms.
* Contribute to the examples or the documentation.

If you don't know where to start, there is a special [Good First
Issue](https://github.com/huggingface/trl/labels/%F0%9F%91%B6%20good%20first%20issue) listing. It will give you a list of
open issues that are beginner-friendly and help you start contributing to open-source. The best way to do that is to open a Pull Request and link it to the issue that you'd like to work on. We try to give priority to opened PRs as we can easily track the progress of the fix, and if the contributor does not have time anymore, someone else can take the PR over.

For something slightly more challenging, you can also take a look at the [Good Second Issue](https://github.com/huggingface/trl/labels/Good%20Second%20Issue) list. In general though, if you feel like you know what you're doing, go for it and we'll help you get there! 🚀

> All contributions are equally valuable to the community. 🥰

Before you start contributing make sure you have installed all the dev tools:

```bash
pip install -e .[dev]
```

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](#submitting-a-pull-request-pr) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature request. It will make it easier for us to come back to you quickly and with good feedback.

### Did you find a bug?

The TRL library is robust and reliable thanks to users who report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the library itself, and not your code.

Once you've confirmed the bug hasn't already been reported, please include the following information in your issue so we can quickly resolve it:

* Your **OS type and version**, **Python**, **PyTorch**, **TRL** and **Transformers** versions.
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s.
* The *full* traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help.

To get the OS and software versions automatically, run the following command:

```bash
trl env
```

### Do you want a new feature?

If there is a new feature you'd like to see in TRL, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it a feature related to something you need for a project? Is it something you worked on and think it could benefit the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.
3. Provide a *code snippet* that demonstrates the feature's usage.
4. If the feature is related to a paper, please include a link.

If your issue is well written we're already 80% of the way there by the time you create it.

## Do you want to implement a new trainer?

New post-training methods are published frequently and those that satisfy the following criteria are good candidates to be integrated into TRL:

* **Simplicity:** Does the new method achieve similar performance as prior methods, but with less complexity? A good example is Direct Preference Optimization (DPO) [[Rafailov et al, 2023]](https://huggingface.co/papers/2305.18290), which provided a simpler and compelling alternative to RLHF methods.
* **Efficiency:** Does the new method provide a significant improvement in training efficiency? A good example is Odds Ratio Preference Optimization (ORPO) [[Hong et al, 2023]](https://huggingface.co/papers/2403.07691), which utilizes a similar objective as DPO but requires half the GPU VRAM.

Methods that only provide incremental improvements at the expense of added complexity or compute costs are unlikely to be included in TRL.

If you want to implement a trainer for a new post-training method, first open an issue and provide the following information:

* A short description of the method and a link to the paper.
* Link to the implementation if it is open-sourced.
* Link to model weights trained with the method if they are available.

Based on the community and maintainer feedback, the next step will be to implement the trainer and config classes. See the following examples for inspiration:

* Paired preference optimisation: [`dpo_trainer.py`](./trl/trainer/dpo_trainer.py) and [`dpo_config.py`](./trl/trainer/dpo_config.py)
* RL-based optimisation: [`rloo_trainer.py](./trl/trainer/rloo_trainer.py) and [`rloo_config.py](./trl/trainer/rloo_config.py)
* Online optimisation: [`online_dpo_trainer.py`](./trl/trainer/online_dpo_trainer.py) and [`online_dpo_config.py`](./trl/trainer/online_dpo_config.py)

## Do you want to add documentation?

We're always looking for improvements to the documentation that make it more clear and accurate. Please let us know how the documentation can be improved, such as typos, dead links, and any missing, unclear, or inaccurate content... We'll be happy to make the changes or help you contribute if you're interested!

## Submitting a pull request (PR)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
TRL. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/huggingface/trl) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote. The following command
   assumes you have your public SSH key uploaded to GitHub. See the following guide for more
   [information](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

   ```bash
   $ git clone git@github.com:<your Github handle>/trl.git
   $ cd trl
   $ git remote add upstream https://github.com/huggingface/trl.git
   ```

3. Create a new branch to hold your development changes, and do this for every new PR you work on.

   Start by synchronizing your `main` branch with the `upstream/main` branch (more details in the [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork)):

   ```bash
   $ git checkout main
   $ git fetch upstream
   $ git merge upstream/main
   ```

   Once your `main` branch is synchronized, create a new branch from it:

   ```bash
   $ git checkout -b a-descriptive-name-for-my-changes
   ```

   **Do not** work on the `main` branch.

4. Set up a development environment by running the following command in a conda or a virtual environment you've created for working on this library:

   ```bash
   $ pip install -e .[dev]
   ```

   (If TRL was already installed in the virtual environment, remove
   it with `pip uninstall trl` before reinstalling it.)

   Alternatively, if you are using [Visual Studio Code](https://code.visualstudio.com/Download), the fastest way to get set up is by using
   the provided Dev Container. Documentation on how to get started with dev containers is available [here](https://code.visualstudio.com/docs/remote/containers).

5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes. You should run the tests impacted by your changes like this (see 
   below an explanation regarding the environment variable):

   ```bash
   $ pytest tests/<TEST_TO_RUN>.py
   ```
   
   > For the following commands leveraging the `make` utility, we recommend using the WSL system when running on
   > Windows. More information [here](https://docs.microsoft.com/en-us/windows/wsl/about).

   You can also run the full suite with the following command.

   ```bash
   $ make test
   ```

    TRL relies on `ruff` for maintaining consistent code formatting across its source files. Before submitting any PR, you should apply automatic style corrections and run code verification checks.

    We provide a `precommit` target in the `Makefile` that simplifies this process by running all required checks and optimizations on only the files modified by your PR.

    To apply these checks and corrections in one step, use:

    ```bash
    $ make precommit
    ```

    This command runs the following:
    - Executes `pre-commit` hooks to automatically fix style issues with `ruff` and other tools.
    - Runs additional scripts such as adding copyright information.

    If you prefer to apply the style corrections separately or review them individually, the `pre-commit` hook will handle the formatting for the files in question.

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   $ git add modified_file.py
   $ git commit
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Push the changes to your account using:

   ```bash
   $ git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors too! To ensure everyone can review your changes in the pull request, work on your local branch and push the updates to your fork. They will automatically appear in the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`, or mark
   the PR as a draft PR. These are useful to avoid duplicated work, and to differentiate
   it from PRs ready to be merged;
4. Make sure existing tests pass;
5. Add high-coverage tests. No quality testing = no merge.


### Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in
the [tests folder](https://github.com/huggingface/trl/tree/main/tests).

We use `pytest` to run the tests. From the root of the
repository here's how to run tests with `pytest` for the library:

```bash
$ python -m pytest -sv ./tests
```

That's how `make test` is implemented (without the `pip install` line)!

You can specify a smaller set of tests to test only the feature
you're working on.

### Default values guidelines

1. **Use defaults when appropriate**:  

Provide default values unless the parameter's value varies significantly by use case. For example, datasets or models should not have defaults, but parameters like `learning_rate` should.

2. **Prioritize proven defaults**:  

Default values should align with those recommended in the original paper or method. Alternatives require strong evidence of superior performance in most cases.

3. **Ensure safety and predictability**:  

Defaults must be safe, expected and reliable. Avoid settings that could lead to surprising outcomes, such as excessive memory usage or poor performance in edge cases.

4. **Balance consistency and flexibility**:  

Aim for consistent defaults across similar functions or methods. However, consistency should not be preferred to point 2 or 3.

5. **Opt-in for new features**:  

Do not enable new features or improvements (e.g., novel loss functions) by default. Users should explicitly opt-in to use these.

### Writing documentation

High-quality documentation is crucial for maintaining a project that is easy to use, understand, and extend. When adding new features, ensure they are thoroughly documented to maintain consistency and clarity throughout the project.

To illustrate what good documentation looks like, here’s an example of a well-documented function:

````python
def replicate_str(string: str, n: int, sep: str = " ") -> str:
    r"""
    Replicate a string `n` times with a separator.

    Args:
        string (`str`):
            String to replicate.
        n (`int`):
            Number of times to replicate the string.
        sep (`str`, *optional*, defaults to `" "`):
            Separator to use between each replication.
    
    Returns:
        `str`: The replicated string.
    
    Examples:
    ```python
    >>> replicate_str("hello", 3)
    "hello hello hello"
    >>> replicate_str("hello", 3, sep=", ")
    "hello, hello, hello"
    ```
    """
    return sep.join([string] * n)
````

* **Line Wrapping:** Applied a consistent line wrap at column 120 to improve readability.
* **Definite Articles:** Removed definite articles where possible to streamline language. (Eg: Changed "The string to replicate" to "String to replicate")
* **Type Annotations:**
  * Always include type definitions, indicating if a parameter is optional and specifying the default value.
  * Note that `Optional` means that the value can be `None`, and `*optional*` means that it is not required for the user to pass a value.
    E.g., for arguments that can't be `None` and aren't required:

    ```python
    foo (`int`, *optional*, defaults to `4`):
    ```

    For arguments that can be `None` and are required:

    ```python
    foo (`Optional[int]`):
    ```

    for arguments that can be `None` and aren't required:

    ```python
    foo (`Optional[int]`, *optional*, defaults to `None`):
    ```

* **String Defaults:**
  * Ensured that default string values are wrapped in double quotes:

    ```python
    defaults to `"foo"`
    ```

* **Dictionary Typing:**
  * Replaced generic `dict` type hints with more explicit `dict[str, Any]` to clarify expected key-value pairs.
* **Default Value Formatting:**
  * Consistently surrounded default values with backticks for improved formatting:

    ```python
    defaults to `4`
    ```

* **Sub-sectioning:** When the number of arguments is large, consider breaking them into sub-sections for better readability.

    ```python
    def calculate_statistics(data: list[float], precision: int = 2, include_variance: bool = False) -> dict[str, float]:
        r"""
        Calculates basic statistics for a given dataset.
    
        Args:
            > Data inputs
    
            data (`list[float]`):
                A list of numerical values to analyze.
    
            > Configuration parameters
    
            precision (`int`, *optional*, defaults to `2`):
                Number of decimal places to round the results.
            include_variance (`bool`, *optional*, defaults to `False`):
                Whether to include the variance of the dataset in the results.
    
        Returns:
            `dict[str, float]`:
                A dictionary containing calculated statistics such as mean, median, and optionally variance.
        """
        ...
      ```

### Deprecation and backward compatibility

Our approach to deprecation and backward compatibility is flexible and based on the feature’s usage and impact. Each deprecation is carefully evaluated, aiming to balance innovation with user needs.

When a feature or component is marked for deprecation, its use will emit a warning message. This warning will include:

- **Transition Guidance**: Instructions on how to migrate to the alternative solution or replacement.
- **Removal Version**: The target version when the feature will be removed, providing users with a clear timeframe to transition.

Example:
   
   ```python
   warnings.warn(
       "The `Trainer.foo` method is deprecated and will be removed in version 0.14.0. "
       "Please use the `Trainer.bar` class instead.",
       FutureWarning,
   )
   ```

The deprecation and removal schedule is based on each feature's usage and impact, with examples at two extremes:

- **Experimental or Low-Use Features**: For a feature that is experimental or has limited usage, backward compatibility may not be maintained between releases. Users should therefore anticipate potential breaking changes from one version to the next.

- **Widely-Used Components**: For a feature with high usage, we aim for a more gradual transition period of approximately **5 months**, generally scheduling deprecation around **5 minor releases** after the initial warning.

These examples represent the two ends of a continuum. The specific timeline for each feature will be determined individually, balancing innovation with user stability needs.

### Working with warnings

Warnings play a critical role in guiding users toward resolving potential issues, but they should be used thoughtfully to avoid unnecessary noise. Unlike logging, which provides informational context or operational details, warnings signal conditions that require attention and action. Overusing warnings can dilute their importance, leading users to ignore them entirely.

#### Definitions

- **Correct**: An operation is correct if it is valid, follows the intended approach, and aligns with the current best practices or guidelines within the codebase. This is the recommended or intended way to perform the operation.
- **Supported**: An operation is supported if it is technically valid and works within the current codebase, but it may not be the most efficient, optimal, or recommended way to perform the task. This includes deprecated features or legacy approaches that still work but may be phased out in the future.

#### Choosing the right message

- **Correct → No warning**:  
   If the operation is fully valid and expected, no message should be issued. The system is working as intended, so no warning is necessary.  

- **Correct but deserves attention → No warning, possibly a log message**:
   When an operation is correct but uncommon or requires special attention, providing an informational message can be helpful. This keeps users informed without implying any issue. If available, use the logger to output this message. Example:  

   ```python
   logger.info("This is an informational message about a rare but correct operation.")
   ```

- **Correct but very likely a mistake → Warning with option to disable**:  
   In rare cases, you may want to issue a warning for a correct operation that’s very likely a mistake. In such cases, you must provide an option to suppress the warning. This can be done with a flag in the function. Example:  

   ```python
   def my_function(foo, bar, _warn=True):
       if foo == bar:
           if _warn:
               warnings.warn("foo and bar are the same, this is likely a mistake. Ignore this warning by setting `_warn=False`.")
           # Do something
   ```

- **Supported but not correct → Warning**:  
   If the operation is technically supported but is deprecated, suboptimal, or could cause future issues (e.g., conflicting arguments), a warning should be raised. This message should be actionable, meaning it must explain how to resolve the issue. Example:  

   ```python
   def my_function(foo, bar):
       if foo and bar:
           warnings.warn("Both `foo` and `bar` were provided, but only one is allowed. Ignoring `foo`. Please pass only one of these arguments.")
           # Do something
   ```

- **Not supported → Exception**:  
   If the operation is invalid or unsupported, raise an exception. This indicates that the operation cannot be performed and requires immediate attention. Example:  

   ```python
   def my_function(foo, bar):
       if foo and bar:
           raise ValueError("Both `foo` and `bar` were provided, but only one is allowed. Please pass only one of these arguments.")
   ```

By following this classification, you ensure that warnings, information, and exceptions are used appropriately, providing clear guidance to the user without cluttering the system with unnecessary messages.
