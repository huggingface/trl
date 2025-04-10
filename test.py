import logging

from datasets import load_dataset

from trl.agent_manager import MultiProcessAider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_to_url(repo_handle: str) -> str:
    return f"https://github.com/{repo_handle}.git"

if __name__ == "__main__":
    ds = load_dataset("princeton-nlp/SWE-bench_Lite")["dev"].select(range(2))
    
    # Create formatted data for the agent - use actual data fields from the dataset 
    data = []
    for example in ds:
        data.append({
            "repo": handle_to_url(example["repo"]),
            "base_commit": example["base_commit"],
            "problem_statement": example["problem_statement"]
        })
    
    logger.info(f"Loaded {len(data)} examples")
    agent = MultiProcessAider()
    logger.info("Agent initialized")

    logger.info("Deploying agent")
    results = agent.deploy(data)
    logger.info("Agent deployed")

    print(results)
