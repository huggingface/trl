# PLAN.md

## Agent Manager Architecture

### Core Concept
The `AgentManager` coordinates ephemeral agents that exist only for the duration of a single training example, replacing `vllm_client.generate()` in the GRPO trainer with an orchestration layer that captures full conversation histories for more effective reinforcement learning.

### Current Implementation

1. **API Middleware Proxy**
   - Lightweight FastAPI server that intercepts API calls between agents and vLLM
   - Injects and tracks conversation via custom `X-Agent-ID` headers
   - Maintains thread-safe conversation history per agent
   - Captures complete request/response pairs for RL training signal

2. **Multiprocessing Approach**
   - Uses `multiprocessing.Pool` for parallel agent execution
   - Each agent runs in an isolated process to prevent state contamination
   - Monkey-patches the `requests` library to inject agent identification headers
   - Process isolation ensures clean environment for each agent instance

3. **Agent Deployment Flow**
   ```
   GRPO Training Step
   └── AgentManager.deploy(prompts)
       ├── Generate unique agent_id for each prompt
       ├── Deploy agents via multiprocessing.Pool
       │   ├── Each process runs _process_one(agent_id, prompt)
       │   │   ├── Monkey-patch requests to add X-Agent-ID
       │   │   └── Call process_one() (e.g., Aider instance)
       │   └── Multiple agents run in parallel
       ├── API Proxy tracks all vLLM interactions
       ├── Await completion with timeout
       ├── Collect conversation histories for all agent_ids
       └── Return structured completions to GRPO trainer
   ```

4. **GRPO Integration**
   - GRPO trainer uses AgentManager.deploy() for generating completions
   - Should properly convert agent completions to token IDs for the training loop
   - Maintains compatibility with both direct vLLM and agent-based generation

## Challenges and Solutions

### Conversation Tracking Challenges

1. **Asynchronous API Calls**
   - Agents make varying numbers of API calls at unpredictable times
   - Solution: Thread-safe conversation tracking with unique agent IDs
   - Thread-safe locking ensures proper history capture even with concurrent requests

2. **Process Management**
   - Challenge: Ensuring clean process termination and resource cleanup
   - Solution: Pool-based multiprocessing with timeout handling
   - Proper cleanup in finally blocks ensures resources are released

3. **Proxy Synchronization**
   - Challenge: Background tasks in FastAPI may create race conditions
   - Solution: Consider making conversation tracking synchronous in the API endpoint
   - More robust synchronization mechanisms for production environments

4. **Conversation Continuity**
   - Challenge: Ensuring continuous context across multiple API calls
   - Solution: Implement validation in the ConversationTracker
   - Track and report potential discontinuities that could indicate information loss

### Technical Considerations

1. **Monkey-Patching Approach**
   - Current: Patch `requests.request` in each worker process to add custom headers
   - Pros: Isolated impact, minimal invasiveness to agent frameworks
   - Alternative: Require direct configuration of agent framework

2. **Conversation Collection**
   - Current: API Proxy collects all conversations by agent_id
   - Challenge: Ensuring all API calls are captured before retrieving history
   - Solution: Consider small delay or synchronization primitive before retrieval

3. **Error Handling**
   - Challenge: Individual agent failures shouldn't crash the entire batch
   - Solution: Improved error handling in AgentManager.deploy()
   - Graceful degradation for failed agents while allowing others to continue

## Conclusions and Next Steps

The current implementation successfully achieves:

1. **Process Isolation**: Clean separation of agent environments
2. **Conversation Tracking**: Complete history capture for RL training
3. **Parallel Execution**: Efficient handling of multiple agents
4. **Resource Management**: Proper cleanup of temporary resources

Next development priorities:

1. **Implement ConversationTracker.get_completion_history()**: Properly extract and format the complete history
2. **Address race conditions**: Ensure background tasks complete before history retrieval
3. **Enhance error handling**: Improve robustness to individual agent failures
4. **Performance optimization**: Evaluate and optimize latency introduced by the proxy
5. **Testing**: Develop comprehensive tests for conversation tracking accuracy