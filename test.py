from datasets import load_dataset

from trl.agent_manager import MultiProcessAider
from trl.git_utils import handle_to_url


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
    
    agent = MultiProcessAider()

    results = agent.deploy(data)

    print(results)
