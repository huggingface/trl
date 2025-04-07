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
       │   ├── Agent 2 
       │   └── ... 
       ├── Monitor progress & conversation history
       ├── Await all completions
       ├── Collect and normalize results
       ├── Clean up resources
       └── Return structured completions
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

# GRPO with Agent Integration

## Current Architecture

The integration of agents with GRPO training follows a "multiprocessing pool" approach for agent deployment:

1. **Agent Manager Base Class**
   - Provides an abstraction layer between GRPO trainer and agent implementations
   - Uses `multiprocessing.Pool` for parallel agent execution
   - Handles process isolation to prevent state contamination between agents
   - Implements timeout mechanism to prevent training stalls

2. **GRPO Trainer Integration**
   - Works alongside vLLM rather than replacing it
   - Handles the distribution of unique prompts across processes
   - Ensures proper synchronization between processes for reward computation
   - Properly converts completions back into token IDs for the training loop

3. **API Spoofing Approach**
   - Agent frameworks (like Aider) typically use OpenAI-compatible API endpoints
   - We redirect these calls to our vLLM server by setting `OPENAI_API_BASE`
   - This allows the model being trained to be used inside the agent scaffolding

## Conversation History Challenges

### Problem Statement

Capturing complete conversation histories from agents presents several critical challenges:

1. **Asynchronous Termination**
   - Different agents may make varying numbers of API calls
   - Some agents complete their tasks quickly while others take much longer
   - Need to know when an agent is truly "done" vs just thinking/processing

2. **Hierarchical Agent Structures**
   - Modern agent frameworks often employ hierarchies of sub-agents
   - Each sub-agent may have its own conversation context and system prompt
   - The "main" conversation might branch into multiple parallel conversations
   - Need to track which responses belong to which conversation thread

3. **Conversation Continuity**
   - Ensuring that each step in a conversation properly builds on previous context
   - Detecting when information is lost or context is reset
   - Handling cases where agents reconstruct or modify past conversation history

4. **Thread Identification**
   - Properly attributing API requests to specific agent instances
   - Maintaining the relationship between requests and responses
   - Ensuring all conversation parts are captured even with concurrent execution

## Proposed Solutions

### Solution 1: API Middleware Proxy

Implement a proxy server that sits between the agent and the vLLM server:

```
Agent Framework → API Proxy → vLLM Server
   (Aider)        (captures)    (generates)
```

**Implementation:**
- Create a lightweight FastAPI server that mimics the OpenAI API
- Add request ID and thread ID headers to track conversation flows
- Maintain an in-memory database of all requests/responses by thread
- After completion, return the full conversation history to the GRPO trainer

**Advantages:**
- Non-invasive to agent frameworks - just change the API endpoint
- Captures all API calls regardless of agent implementation details
- Preserves the complete call sequence and timestamps

**Challenges:**
- Extra network hop adds latency
- Maintaining the proxy adds complexity
- Need to handle connection failures

### Solution 2: Conversation State Manager

Create a centralized conversation state manager to track all API interactions:

**Implementation:**
- Use process-local variable to store an agent ID in each worker process
- Monkey-patch the `requests` library in agent processes to intercept API calls
- Implement callbacks to send conversation data back to a central service
- Maintain a mapping of agent ID → conversation history

**Advantages:**
- No network overhead
- Works with any agent that uses standard HTTP libraries
- Can handle hierarchical agents by tracking conversation threads

**Challenges:**
- More invasive modification to the execution environment
- May not work with all agent implementations
- Requires careful cleanup to prevent memory leaks

### Solution 3: Post-Processing Approach

Instead of trying to track the conversation in real-time, extract it from the completed agent:

**Implementation:**
- Allow agents to run to completion normally
- After completion, ask the agent to export its full conversation history
- For agent frameworks that don't support history export, scrape logs or memory
- Reconstruct token sequences from the conversation history

**Advantages:**
- Simplest implementation
- Most compatible with varied agent implementations
- Allows natural agent operation without interference

**Challenges:**
- Relies on agent's ability to export its history
- May miss some details of the conversation
- Less control during execution

## Recommended Approach

A hybrid approach combining elements from Solutions 1 and 3:

1. Use a lightweight API middleware proxy to track base API calls
2. Allow for agent-specific history export methods as an enhancement
3. Implement detection for conversation discontinuities
4. Use a timeout mechanism to handle agents that don't terminate cleanly

This approach provides the best balance of:
- Robust conversation capture
- Compatibility with different agent frameworks
- Minimal interference with agent operation
- Clean integration with GRPO training

## Next Steps

1. Implement the API middleware proxy
2. Add conversation history extraction functionality to the AgentManager
3. Enhance the GRPO trainer to properly handle conversation-based completions
4. Add support for agent-specific reward calculation based on conversation quality
5. Implement robust error handling and timeouts for agent processes