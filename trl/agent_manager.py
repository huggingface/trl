import os
import logging
import multiprocessing
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from contextlib import redirect_stdout, redirect_stderr

import requests

from trl.import_utils import is_aider_available
from trl.git_utils import clone_repo_at_commit, clean_repo_dir

if is_aider_available():
    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model


logger = logging.getLogger(__name__)


class AgentManager(ABC):
    """
    Base class for agent managers that coordinate ephemeral agents during GRPO training.

    The AgentManager replaces vllm_client.generate() in the GRPO trainer by deploying
    agents that process prompts and return completions. Each agent exists only for the
    duration of a single training example.
    """

    def __init__(self, vllm_url: str = "http://localhost:8000/v1"):
        self.vllm_url = vllm_url

        os.environ["OPENAI_API_BASE"] = self.vllm_url
        os.environ["OPENAI_API_KEY"] = "dummy-key"

    def match_histories_to_prompts(self, histories: List[List[Dict[str, str]]], prompts: List[str]) -> List[List[Dict[str, str]]]:
        """Uses the ordering of the prompts to find a history that matches the prompt order"""
        return histories  # TODO: Implement this

    @abstractmethod
    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[List[Dict[str, str]]]:
        """
        Deploy parallel agents to process the given prompts, returning histories.
        """
        ...


class MultiProcessAider(AgentManager):
    """Agent manager that uses multiple processes to parallelize agent deployments."""

    def __init__(self, vllm_url: str = "http://localhost:8000/v1"):
        """
        Initialize the agent manager.

        Args:
            vllm_url: URL of the vLLM server (including protocol, host and port)
        """
        super().__init__(vllm_url)

        if not is_aider_available():
            raise ImportError("Aider is not installed. Please install it with `pip install aider`.")

    def process_one(self, prompt: Dict[str, Any]):
        """Process a single prompt and return a completion"""

        assert ("repo" in prompt and "base_commit" in prompt and "problem_statement" in prompt), "Data should contain repo, base_commit and problem_statement"

        temp_folder = None  # Initialize to avoid UnboundLocalError
        original_dir = os.getcwd()  # Save current directory

        try:
            # Clone the repo into a temporary folder
            temp_folder = clone_repo_at_commit(prompt["repo"], prompt["base_commit"])

            # Change to the repo's root directory so Aider can compute the repo-map
            os.chdir(temp_folder)

            # Redirect Aider's terminal output to the void
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
                coder = Coder.create(
                    main_model=Model("openai/Qwen2.5-Coder-32B-Instruct"),  # Just a placeholder, the spoofed OPENAI_API_BASE will make it use vLLM
                    io=InputOutput(yes=True),
                )
                coder.run(prompt["problem_statement"])

        finally:
            if temp_folder:
                clean_repo_dir(temp_folder)
            os.chdir(original_dir)

    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[List[Dict[str, str]]]:
        """
        Deploy parallel agents to process the given prompts, returning histories.

        Args:
            prompts: List of prompts to process
            timeout: Maximum time in seconds to wait for all prompts to complete

        Returns:
            List of unordered results
        """
        # Process all prompts in parallel
        with multiprocessing.Pool() as pool:
            # Start async processing of all prompts
            result = pool.map_async(self.process_one, prompts)

            try:
                # Wait for results with timeout
                result.get(timeout=timeout)
            except multiprocessing.TimeoutError:
                logger.warning(f"Agent timeout reached after {timeout} seconds.")

        # get histories from vllm
        histories = requests.get(f"{self.vllm_url}/histories")
        print(histories)


class ApptainerAider(AgentManager):
    """Agent manager that uses apptainer containers to parallelize agent deployments."""

    def __init__(self, vllm_url: str = "http://localhost:8000", apptainer_image: str = "aider.sif"):
        """
        Initialize the agent manager.

        Args:
            vllm_url: URL of the vLLM server (including protocol, host and port)
            apptainer_image: Path to the apptainer image
        """
        super().__init__(vllm_url)

        self.apptainer_image = apptainer_image

        if not is_aider_available():
            raise ImportError("Aider is not installed. Please install it with `pip install aider`.")

    def deploy(self, prompts: List[Dict[str, Any]], timeout: int = 300) -> List[List[Dict[str, str]]]:
        pass
