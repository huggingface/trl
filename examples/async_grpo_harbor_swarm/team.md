Solve the escape-room puzzle with your teammates. Each agent has an independent copy:
share discoveries, but reproduce prerequisite actions in your own room. The team stops
after the first verified win. Start solving immediately; do not wait for a coordinator.

`escape` and `flock-tool` are shell commands: invoke them through your bash tool.
Start with `escape observe`, then introduce yourself:

```sh
flock-tool message_board.whoami
flock-tool message_board.post --json '{"channel":"general","text":"I am starting exploration."}'
flock-tool message_board.read --json '{"channel":"general","after":0,"limit":12}'
```

Claim a branch or node, inspect its interface, and share verified findings, useful failed
inputs, and specific requests for help. Keep posts under 500 characters. Preserve exact
arguments, values, and newlines. Use `--json -` to supply JSON on stdin when quoting is awkward.
Do not share credentials or private verifier artifacts.

After every three puzzle actions, read general again. Set `after` to the highest channel
`seq` actually returned by your previous read. Never advance it from your own post response.
Keep it unchanged after empty or failed reads; read another page when a full page is returned.
Do not busy-poll. Acknowledge useful teammate discoveries in your next substantive update.

Inspect entities before calling them. Use `escape tool-call` or `escape use-item` with the
actual node IDs and all required parameters. Preserve initial values exactly. Repeating
prerequisites in your own room is expected; repeating unchanged failed guesses is not useful.

The victory node ID is not the answer. Obtain its output and submit it unchanged with
`escape submit-answer`. Only a successful submission confirms a win. Post the verified win
and finish; otherwise continue until your action, model-turn, or time budget ends.
