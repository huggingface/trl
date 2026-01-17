# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import time
from collections.abc import Callable

from transformers import Trainer
from transformers.integrations import is_mlflow_available, is_wandb_available


if is_wandb_available():
    import wandb

if is_mlflow_available():
    import mlflow


class ProfilingContext:
    """
    Context manager for profiling code blocks with configurable logging.

    This class handles timing of code execution and logging metrics to various backends (Weights & Biases, MLflow)
    without being coupled to the Trainer class.

    Args:
        name (`str`):
            Name of the profiling context. Used in the metric name.
        report_to (`list` of `str`):
            List of integrations to report metrics to (e.g., ["wandb", "mlflow"]).
        is_main_process (`bool`, *optional*, defaults to `True`):
            Whether this is the main process in distributed training. Metrics are only logged from the main process.
        step (`int` or `None`, *optional*):
            Training step to associate with the logged metrics.
        metric_prefix (`str`, *optional*, defaults to `"profiling/Time taken"`):
            Prefix for the metric name in logs.

    Example:
    ```python
    # Direct usage
    from trl.extras.profiling import ProfilingContext

    with ProfilingContext(
        name="MyClass.expensive_operation",
        report_to=["wandb"],
        is_main_process=True,
        step=100,
    ):
        # Code to profile
        result = expensive_computation()

    # With Trainer (backwards compatible via profiling_context function)
    from transformers import Trainer
    from trl.extras.profiling import profiling_context


    class MyTrainer(Trainer):
        def some_method(self):
            with profiling_context(self, "matrix_multiplication"):
                result = matrix_multiply()
    ```
    """

    def __init__(
        self,
        name: str,
        report_to: list[str],
        is_main_process: bool = True,
        step: int | None = None,
        metric_prefix: str = "profiling/Time taken",
    ):
        self.name = name
        self.report_to = report_to
        self.is_main_process = is_main_process
        self.step = step
        self.metric_prefix = metric_prefix
        self._start_time = None

    def __enter__(self):
        """Start timing when entering the context."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log metrics when exiting the context."""
        if self._start_time is not None:
            duration = time.perf_counter() - self._start_time
            self._log_metrics(duration)
        return False

    def _log_metrics(self, duration: float) -> None:
        """
        Log profiling metrics to configured backends.

        Args:
            duration (`float`):
                Execution time in seconds.
        """
        if not self.is_main_process:
            return

        metric_name = f"{self.metric_prefix}: {self.name}"
        metrics = {metric_name: duration}

        # Log to Weights & Biases if configured
        if "wandb" in self.report_to and is_wandb_available() and wandb.run is not None:
            wandb.log(metrics)

        # Log to MLflow if configured
        if "mlflow" in self.report_to and is_mlflow_available() and mlflow.active_run() is not None:
            mlflow.log_metrics(metrics, step=self.step)


def profiling_context(trainer: Trainer, name: str) -> ProfilingContext:
    """
    Factory function to create a ProfilingContext from a Trainer instance.

    This function maintains backwards compatibility with existing code while using the decoupled ProfilingContext class
    internally.

    Args:
        trainer (`~transformers.Trainer`):
            Trainer object containing configuration for logging.
        name (`str`):
            Name of the block to be profiled. Will be prefixed with the trainer class name.

    Returns:
        `ProfilingContext`: A configured profiling context manager.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_context


    class MyTrainer(Trainer):
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            with profiling_context(self, "matrix_multiplication"):
                # Code to profile: simulate a computationally expensive operation
                result = A @ B  # Matrix multiplication
    ```
    """
    context_name = f"{trainer.__class__.__name__}.{name}"
    step = trainer.state.global_step

    return ProfilingContext(
        name=context_name,
        report_to=trainer.args.report_to,
        is_main_process=trainer.accelerator.is_main_process,
        step=step,
    )


def profiling_decorator(func: Callable) -> Callable:
    """
    Decorator to profile a function and log execution time using [`extras.profiling.profiling_context`].

    This decorator works with methods that have access to a trainer instance (typically as `self`).

    Args:
        func (`Callable`):
            Function to be profiled.

    Returns:
        `Callable`: Wrapped function that profiles execution time.

    Example:
    ```python
    from transformers import Trainer
    from trl.extras.profiling import profiling_decorator


    class MyTrainer(Trainer):
        @profiling_decorator
        def some_method(self):
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            # Code to profile: simulate a computationally expensive operation
            result = A @ B
    ```
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        with profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper
