# Adapters

An adapter connects runthrough to the agent you're testing. It sends the conversation so far
and returns the agent's reply as an `AgentTurn`: text, tool calls and tool results, split into
steps where the protocol shows them.

## Vercel AI SDK (`ai_sdk_endpoint`)

For agents built with the Vercel AI SDK v5 that stream the
[UI-message protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol), i.e. any backend
a `useChat` frontend talks to (`toUIMessageStreamResponse()` or
`pipeUIMessageStreamToResponse()`).

```python
from runthrough import ai_sdk_endpoint

agent = ai_sdk_endpoint(
    "http://localhost:3000/api/chat",
    headers={"x-api-key": os.environ["AGENT_API_KEY"]},
)
```

### What gets sent

Each turn, runthrough POSTs what `useChat` would send:

```json
{
  "id": "<conversation id, a UUID, new for every sample and epoch>",
  "messages": [ <the whole conversation as UIMessages> ],
  "trigger": "submit-message",
  "...": "the scenario's context fields"
}
```

- Earlier agent replies are resent as assistant UIMessages with `step-start`, `text` and
  `tool-<name>` parts (state `output-available` or `output-error`), so the agent sees the same
  history a browser would send.
- The scenario's `context` is merged into the top level of the body, like `useChat`'s `body`
  option. Pass `body={...}` or `body=lambda ctx: {...}` to send something else instead.
- `headers` can also be a function of the turn context, e.g. to add a per-run id.

### What gets read

| Stream event | Becomes |
|---|---|
| `text-start` / `text-delta` / `text-end` | Text of the current step |
| `tool-input-available`, `tool-input-error` | A tool call (with its arguments) in the current step |
| `tool-output-available` / `tool-output-error` | That call's result or error |
| `start-step` / `finish-step` | Step boundaries |
| `start` | The agent's message id |
| `error`, `abort` | `AgentTurn.error`; the conversation ends with `agent_error` |

Deltas for reasoning and tool input are folded away; other events are kept in
`AgentTurn.raw_events`.

### Errors

| Response | Result |
|---|---|
| 5xx | Agent error: recorded, the conversation ends, scorers can count it as a failure. |
| 4xx | `AdapterError`: the sample fails. Usually a wrong URL or API key. |
| Not `text/event-stream`, or an event that isn't JSON | `AdapterError` with the start of the body, to help find the wrong route. |
| Connection error or timeout | The sample fails; use Inspect's `--retry-on-error`. |

### Test fixtures

`tests/fixtures/ai_sdk/*.sse` are real streams recorded from `ai` v5 with a mock model
(`tests/fixtures/ai_sdk/generate.mjs`), so parsing is tested against what the SDK actually
emits.

## Your own adapter

Anything with a `history` attribute and an async `send()` method works:

```python
from runthrough import AgentTurn, TurnContext


class MyAgent:
    history = "session"  # the agent keeps history; send only the new message

    async def send(self, messages, ctx: TurnContext) -> AgentTurn:
        reply = await my_client.chat(ctx.conversation_id, messages[-1].text)
        return AgentTurn(text=reply.text)
```
