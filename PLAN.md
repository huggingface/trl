# PLAN.md

## Agent Manager Architecture

### Core Concept
The `AgentManager` will serve as a coordinator for ephemeral agents that exist only for the duration of a single training example. It replaces the `vllm_client.generate()` method in the GRPO trainer with a more sophisticated orchestration layer.

### Implementation Structure

1. **Base AgentManager Class**
   - Provides the interface and shared functionality
   - Handles communication with the vLLM server
   - Coordinates multiple parallel agents
   - Collects and processes results

2. **Agent Deployment**
   - `deploy()` method will:
     - Initialize required environments for agents (e.g., temp repos)
     - Launch N agents in parallel (potentially different agent types)
     - Await completion from all agents
     - Collect completion histories
     - Clean up resources
     - Return standardized completion IDs

3. **Agent Communication**
   - Each agent gets its own temporary endpoint
   - Endpoints proxy to the central vLLM server
   - Each request/response is recorded for history tracking
   - Standardized interface ensures compatibility with GRPO trainer

4. **Agent Lifecycle**
   ```
   GRPO Training Step
   └── AgentManager.deploy()
       ├── Create ephemeral resources (repos, endpoints)
       ├── Deploy N parallel agents
       │   ├── Agent 1 (e.g., Aider instance)
       │   ├── Agent 2 (different approach)
       │   └── ... 
       ├── Monitor progress & conversation history
       ├── Await all completions
       ├── Collect and normalize results
       ├── Clean up resources
       └── Return structured completions
   ```

## Implementation Details

```python
class AgentManager(ABC):
    def __init__(self, vllm_port=8000):
        self.vllm_port = vllm_port
        self.endpoints = {}  # Tracks active endpoints

    @abstractmethod
    def deploy(self, data: Dict[str, Any]) -> List[str]:
        """Deploy agents to process the given data, returning completion IDs"""
        pass
        
    def _create_endpoint(self, agent_id):
        """Creates a unique endpoint for an agent that records history"""
        port = self._allocate_port()
        # Setup FastAPI endpoint that proxies to vLLM and records history
        # ...
        return port
        
    def _cleanup(self):
        """Release all resources used by agents"""
        # Stop all endpoints, remove temp directories, etc.
```

## Specialized Implementations

```python
class AiderAgentManager(AgentManager):
    def deploy(self, data: Dict[str, Any]) -> List[str]:
        # Clone repos into temporary directories
        repo_paths = self._clone_repos(data)
        
        # Initialize Aider instances, each with its own endpoint
        agents = []
        for repo_path in repo_paths:
            agent_id = str(uuid.uuid4())
            port = self._create_endpoint(agent_id)
            agent = self._create_aider_agent(repo_path, port)
            agents.append(agent)
            
        # Run all agents in parallel
        completion_futures = [self._run_agent(agent) for agent in agents]
        
        # Wait for all to complete
        completions = await asyncio.gather(*completion_futures)
        
        # Clean up
        self._cleanup()
        
        return completions
```

## Concerns and Potential Issues

### Context Continuity
- **Problem**: Agents may dynamically construct their context or operate in hierarchies
- **Detection**: Compare successive requests to ensure context T is included in T+1
- **Solution**: Implement warning system for context discontinuities that might indicate information loss

### Synchronization
- **Problem**: Different agents may complete at vastly different times
- **Solution**: Use asyncio to manage parallel execution without blocking

### Resource Management
- **Problem**: Temporary endpoints and environments must be properly cleaned up
- **Solution**: Implement robust cleanup with try/finally blocks and resource tracking

### Standardization
- **Problem**: Different agent types produce different output formats
- **Solution**: Implement normalization layer to convert all agent outputs to GRPO-compatible format

### Performance
- **Problem**: Managing multiple agents could introduce overhead
- **Solution**: Use lightweight proxying and ensure vLLM's dynamic batching capabilities are utilized

### Error Handling
- **Problem**: Agent failures should not crash the entire training process
- **Solution**: Implement graceful degradation and robust error reporting

### Training Signal Quality
- **Problem**: Different agent approaches may produce inconsistent training signals
- **Solution**: Consider normalization techniques for rewards across different agent types