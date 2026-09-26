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


def test_server_tool_continuations_keep_their_own_histories(server_tool_trainer):
    """Two siblings with different tool results must each continue from their own history."""
    trainer = server_tool_trainer
    # The same initial prompt produced two different assistant tool calls (tokens 10 and 20), returning 30 and 35.
    # The mock server returns the last token + 100, so these histories should receive 130 and 135 respectively.
    prompts = [[{"role": "user", "content": "Calculate 3 * 10 + 5."}] for _ in range(2)]
    completions = [
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "calculator", "arguments": {"result": result}}}
                ],
            }
        ]
        for result in (30, 35)
    ]

    tool_mask, messages, completion_ids, logprobs, tool_count, failure_count, _ = trainer._tool_call_loop(
        prompts=prompts,
        prompt_ids=[[1], [1]],
        completion_ids=[[10], [20]],
        completions=completions,
        logprobs=[[-0.1], [-0.2]],
        images=None,
        multimodal_fields={},
    )

    assert tool_count == 2, "Both siblings must execute their calculator call before continuing."
    assert failure_count == 0, "Calculator calls must succeed for this ownership test."
    assert [history[1]["content"] for history in messages] == ["30", "35"], (
        "Each history must retain its own calculator result: 30 for the first sibling, 35 for the second."
    )
    assert completion_ids == [[10, 30, 130], [20, 35, 135]], (
        "Continuations must use their own tool history: the second sibling must receive token 135; "
        "token 130 means its answer was generated from the first sibling's history."
    )
    # Distinct mock logprobs let us verify probability ownership as well as token ownership.
    assert logprobs == [[-0.1, 0.0, -0.3], [-0.2, 0.0, -0.35]], (
        "Continuation logprobs must stay with their originating history: -0.3 for the first sibling, "
        "-0.35 for the second; tool-result positions must have zero placeholders."
    )
    assert tool_mask == [[1, 0, 1], [1, 0, 1]], "Only tool-result tokens should be excluded from the training loss."
    assert [history[-1]["content"] for history in messages] == ["130", "135"], (
        "Decoded assistant messages must preserve the same sibling ownership as the completion tokens."
    )
    request = trainer.vllm_generation.vllm_client.generate.call_args.kwargs
    assert request["prompts"] == [[1, 10, 30], [1, 20, 35]], (
        "The server must receive both distinct continuation histories in their original order."
    )
    assert request["n"] == 1, "Each continuation history must request exactly one completion."
