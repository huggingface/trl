import logging

from datasets import load_dataset

from trl.agent_manager import MultiProcessAider
from trl.git_utils import handle_to_url

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    logger.info("Starting test")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite")["dev"].select(range(2))
    logger.info(f"Loaded dataset with {len(ds)} examples")
    
    # Create formatted data for the agent - use actual data fields from the dataset 
    data = []
    for example in ds:
        data.append({
            "repo": handle_to_url(example["repo"]),
            "base_commit": example["base_commit"],
            "problem_statement": example["problem_statement"]
        })
    
    logger.info("Initializing agent manager")

    agent = MultiProcessAider()
    logger.info("Agent manager initialized")

    logger.info("Deploying agent")
    results = agent.deploy(data)
    logger.info("Agent deployed")

    logger.info(f"Results: {results}")
