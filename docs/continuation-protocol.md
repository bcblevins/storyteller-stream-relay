# Assistant continuation protocol

Send `tool_protocol: "continuation_v1"` to `/v1/stream`, with the existing JWT,
explicit bot ID, app-assembled messages, and optional tools. Each request executes
one provider round. The relay never executes tools or persists application data.
Requests without this field keep their old behavior. Unknown versions return 400.
Opt-in requests own separate provider clients, closed with their streams, and
disable SDK automatic retries. The app owns explicit, recorded retry attempts.

```json
{
  "tool_protocol": "continuation_v1",
  "bot_id": 9,
  "messages": [{"role": "user", "content": "Find the harbor."}],
  "tools": [{"type": "function", "function": {
    "name": "world_index", "parameters": {"type": "object"}
  }}]
}
```

Tools are optional: text-only Storyteller/Synthesis rounds can use the same
protocol for reliable finish metadata. No-tool requests omit tool parameters
upstream. With tools, the relay requests `parallel_tool_calls: false` but preserves
multiple calls if a provider nevertheless returns them.

## SSE contract

`token`, `reasoning`, and `tool_call_start` are progress only. This protocol does
not emit legacy `tool_call` events. Execute nothing before the successful terminal
envelope; it is authoritative for content and complete calls.

```json
{
  "stream_id": "example",
  "tool_protocol": "continuation_v1",
  "status": "tool_calls",
  "assistant_message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [{"id": "call_1", "type": "function", "function": {
      "name": "world_index", "arguments": "{\"operation\":\"list\"}"
    }}]
  },
  "finish_reason": "tool_calls",
  "usage": null,
  "model": "resolved-request-model",
  "provider_model": null
}
```

The `done` event above has `status: "tool_calls"` when calls exist, otherwise
`status: "completed"` and `finish_reason: "stop"`. `model` is the resolved model
sent to the provider; `provider_model` is its reported model when available.
Usage is the terminal provider report, never an estimate or a sum of repeated
events. Missing usage/model reporting is null. No credentials enter this envelope.

Append `assistant_message` unchanged, then one `role: "tool"` message per call
with the same `tool_call_id` and string `content`. Send the tools again on the next
request. Do not rebuild assistant content from display text or parse/re-serialize
the raw `function.arguments` string in the saved assistant message.

## Continuation fidelity and validation

The relay concatenates textual `reasoning_content` and `reasoning` fragments.
It preserves the received ordered sequence of `reasoning_details` objects,
including nulls, encrypted data, and signatures. They remain machine protocol
data, separate from the human-readable reasoning progress event. It does not
invent or summarize provider continuation state.

Call fragments are combined by index; IDs/names are fixed and argument fragments
concatenate. Missing/conflicting identities, duplicate IDs, invalid JSON syntax,
nonstandard JSON constants, refusal, missing finish reasons, truncated/filtered
responses, and unexpected data after finish produce one `error`, not `done`.
Error metadata includes protocol/stream IDs, finish reason, known usage, and a
`partial_assistant_message` for inspection only. It has no executable
`assistant_message` envelope. Cancellation/disconnect produces no terminal
success and closes upstream resources.

Valid JSON is not sufficient permission to execute. The client must additionally
reject non-object or schema-invalid arguments, unknown/unavailable tools, and
exhausted budgets. The relay intentionally does not enforce product tool schemas.

This supports the existing Chat Completions-compatible transport, including the
documented DeepSeek/OpenRouter continuation fields. It is not a native Anthropic
or Responses adapter and does not promise every provider/model works. Validate
chosen models with live smoke tests before enabling them in a product. No live
provider requests or deployment are implied by mocked protocol tests.

## Verification

Install the pinned `requirements.txt` using Python 3.11+ in an isolated environment.
The suite uses mocked providers. Supply dummy required settings, not credentials:

```sh
env SUPABASE_URL=https://example.invalid SUPABASE_JWT_SECRET=test-only \
  SUPABASE_REST_URL=https://example.invalid/rest/v1 SUPABASE_ANON_KEY=test-only \
  OPENROUTER_PROVISIONING_KEY=test-only OPENROUTER_DEMO_MODEL=test-model \
  OPENROUTER_DEMO_LIMIT=1 python -m unittest discover -s tests -q
```

Tests cover fragmentation, exact continuation replay, multiple calls, malformed
streams, terminal metadata, opt-in routing, closure on cancellation/disconnect,
overlapping request isolation, retry settings, and legacy-path regressions.
Lead verification: 61 tests passed on Python 3.12.14 with pinned dependencies.

Primary references:

- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenRouter tool calling](https://openrouter.ai/docs/guides/features/tool-calling)
- [OpenRouter reasoning details](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)
