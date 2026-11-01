# Storyteller Relay

The **Storyteller Relay** is a specialized backend service designed to handle real-time AI streaming and authentication for the Storyteller application.

It serves as the secure "switchboard" between the client frontend, the database (Supabase), and Large Language Model providers (OpenAI/DeepSeek), ensuring that sensitive API keys remain server-side while delivering low-latency token streaming to users.

## 🎯 Why This Exists

Directly connecting a frontend to an LLM provider exposes API keys and duplicates provider-specific streaming logic. This relay solves those problems by:

1.  **Securing Credentials:** It holds the LLM API keys, so the frontend never sees them.
2.  **Centralizing Auth:** It verifies Supabase JWTs before processing any request.
3.  **Relaying Tokens:** It streams model output to authenticated clients with low latency.
4.  **Keeping Message Ownership Client-Side:** The frontend owns conversation message persistence, rerolls, alternatives, and cleanup.

## 🏗️ Architecture

The service is built with **FastAPI** and designed to be stateless and scalable.

### Core Flows

* **Streaming Pipeline:**
    * **Input:** Receives a user message context + system prompt + Supabase Auth Token.
    * **Process:** Authenticates the user, fetches the appropriate "Bot" configuration (model, temperature, system instructions) from Supabase, and opens a stream to the LLM provider.
    * **Output:** Streams tokens back to the client via Server-Sent Events (SSE).
    * **Cleanup:** On completion it emits a terminal SSE event. The relay does not persist conversation messages or partial output.

* **Bot Resolution Strategy:**
    The relay dynamically decides which AI persona to use for a response based on a priority hierarchy:
    1.  **Explicit:** A specific `bot_id` passed in the payload.
    2.  **Conversation-Bound:** The bot assigned to the specific conversation ID.
    3.  **Default:** The user's preferred default bot.
    4.  **Fallback:** The most recently updated bot owned by the user.

### Key Features

* **Server-Sent Events (SSE):** Uses `sse_starlette` for efficient, real-time text streaming.
* **Frontend-Owned Conversation Persistence:** `/v1/stream` builds prompts from request-body `messages` and never reads or writes conversation message rows.
* **Rate Limiting:** Includes basic in-memory rate limiting to prevent abuse.
* **CORS Management:** specialized handling to support secure cross-origin requests from the Storyteller frontend.

## Client-owned tool loops

All application generations use `POST /v1/stream`. The relay executes one provider
round; the app executes tools, appends results, and requests further rounds.
The former Creator-specific endpoints have been removed.

Opt in with `tool_protocol: "continuation_v1"` for a complete assistant message in
the terminal `done` event, including ordered tool calls and provider-required
continuation data. This also works for text-only requests. See the
[protocol contract](docs/continuation-protocol.md) for payloads, failure behavior,
compatibility limits, and test instructions.

Requests without the opt-in retain the existing `token`, `reasoning`,
`tool_call_start`, `tool_call`, and terminal event formats. The legacy tool path
still allows only one returned call and reports `awaiting_tool_approval`; new
callers should use the protocol above. No path writes application data.

## 🛠️ Tech Stack

* **Framework:** Python FastAPI
* **Streaming:** `sse-starlette`
* **Database/Auth:** Supabase (REST API & JWT Verification)
* **LLM Integration:** OpenAI SDK (compatible with DeepSeek and other OpenAI-like endpoints)
* **Runtime:** Python 3.11+

## GLM Addon: OpenRouter Chat Completions Proxy

This relay now includes an optional addon endpoint for your personal GLM usage:

* `POST /v1/chat/completions`

It is isolated from the main Storyteller streaming flow (`/v1/stream`) and uses its own API-key gate.

### Security and Access

The addon endpoint requires:

* `Authorization: Bearer <GLM_PROXY_API_KEY>`

The same key is then used upstream with OpenRouter. If `GLM_PROXY_API_KEY` is not set, the endpoint returns `503` and remains unavailable.

### Reasoning and Prompt Controls

The relay enables provider-specific thinking/reasoning defaults server-side and preserves explicit request settings unless override is enabled. These controls apply to the addon proxy and the main conversation stream (`POST /v1/stream`); creator streaming remains unchanged.

OpenRouter requests get:

* `reasoning.enabled = true`

If `FORCE_REASONING_EFFORT` is set, that value is also attached where supported. Other recognized providers use their native conventions (`reasoning_effort` for OpenAI reasoning models, `thinking` bodies for Anthropic-compatible and DeepSeek-compatible endpoints).

### Environment Variables

* `GLM_PROXY_API_KEY` (default: unset)
* `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
* `FORCE_REASONING_ENABLED` (default: `true`)
* `FORCE_REASONING_EFFORT` (default: unset)
* `FORCE_REASONING_MODEL_PATTERNS` (default: `*`) comma-separated glob patterns
* `FORCE_REASONING_OVERRIDE` (default: `false`)
* `ENABLE_SYSTEM_INJECTION_TAG` (default: `true`)
* `SYSTEM_INJECTION_TAG_NAME` (default: `injection`)
* `ENABLE_SYSTEM_THINKING_TAG` (default: `true`)
* `SYSTEM_THINKING_TAG_NAME` (default: `thinking`)

Example:

```bash
export GLM_PROXY_API_KEY="sk-or-..."
export FORCE_REASONING_ENABLED="true"
export FORCE_REASONING_EFFORT=""
export FORCE_REASONING_MODEL_PATTERNS="*"
export FORCE_REASONING_OVERRIDE="false"
export ENABLE_SYSTEM_INJECTION_TAG="true"
export SYSTEM_INJECTION_TAG_NAME="injection"
export ENABLE_SYSTEM_THINKING_TAG="true"
export SYSTEM_THINKING_TAG_NAME="thinking"
```

When `ENABLE_SYSTEM_INJECTION_TAG=true`, supported conversation requests scan system messages for
`<injection>...</injection>` (or your configured tag), removes those blocks from
system content, and appends the extracted text to the latest message before
sending the provider request.

When `ENABLE_SYSTEM_THINKING_TAG=true`, supported conversation requests scan system messages for
`<thinking>...</thinking>` (or your configured tag), removes those blocks from
system content, and maps the extracted control to provider reasoning settings.
Supported values are `enabled`, `disabled`, effort-only values like `high` or
`max`, and combined values like `enabled:max`. For OpenRouter this maps to the
`reasoning` object; for DeepSeek-compatible requests this maps to
`extra_body.thinking` plus `reasoning_effort` when an effort is supplied.

### Usage Example

```bash
curl -N \
  -H "Authorization: Bearer $GLM_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-ai/glm-4.6:nitro",
    "stream": true,
    "messages": [{"role":"user","content":"Give me a short plan."}]
  }' \
  http://localhost:8000/v1/chat/completions
```

Note: forcing reasoning can increase output tokens, latency, and cost.


## External mock chat completions

`POST /mock/v1/chat/completions` returns the fixed assistant text
`This is a mock streamed response.` without calling a model or Supabase. It uses
the same `Authorization: Bearer <GLM_PROXY_API_KEY>` authentication as the external
proxy and returns `503` if that key is not configured.

For OpenAI-compatible clients, set the base URL to `<relay-host>/mock/v1` and use
the usual chat-completions method. The request may include normal `model` and
`messages` fields; the response echoes the model (default `mock-model`). Messages
and generation controls are ignored, including prompt-capture markers. No
request transforms run, no usage counts are generated, and no data is persisted.

With `stream: true`, the endpoint immediately sends OpenAI-compatible SSE
`data:` frames: an assistant-role chunk, word-sized content chunks, a final
chunk with `finish_reason: "stop"`, and `data: [DONE]`. There are no artificial
delays. With `stream` omitted or false, it returns ordinary chat-completion JSON
containing the same text.

```bash
curl -N \
  -H "Authorization: Bearer $GLM_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mock-model","stream":true,"messages":[{"role":"user","content":"Hello"}]}' \
  http://localhost:8000/mock/v1/chat/completions
```

## External system extensions

On `POST /v1/chat/completions`, the first complete, case-sensitive
`<system-extension>...</system-extension>` block in message/text-part order is
removed from its source and appended to the last system message, separated by a
blank line. Multiline contents and surrounding text are preserved. Multipart
system content receives a new text part at its end. If there is no system
message, a new one is prepended. The original request object is not mutated.

Only that first block is consumed; later occurrences stay unchanged. An empty
or whitespace-only first block is removed and consumes the opportunity, without
appending instructions or creating a system message. Blocks must be wholly in a
single string or standard text content part; attributes, nested extension tags,
and cross-part blocks are unsupported. An unclosed block is left untouched.
Unsupported last-system content is left unchanged rather than overwritten.

The external transformation order is system extension, capture, injection,
thinking, then provider mapping. Captures therefore include the extended system
text, and an optional capture `user` replacement applies to the extension too.
Existing injection and thinking behavior runs afterward. Application `/v1/stream` requests do not process system extensions.

## External system prompt capture

The separate `POST /v1/chat/completions` proxy recognizes `<capture-content/>`
(or `<capture-content />`) in any message's text, including standard text content
parts. It removes all markers and attempts one capture of system text after
system extension and before injection/thinking transforms. Other message text
is not stored unless explicitly promoted by a system extension. Main application `/v1/stream` requests do not capture.

An optional `user` attribute, e.g. `<capture-content user="Eryndor"/>`, replaces
every literal, case-sensitive occurrence of that name in captured system text
with `{{user}}`. Names in forwarded messages stay unchanged; only capture tags
are removed. Single or double quotes and whitespace around `=` are supported.
With multiple markers, the first nonempty `user` value in message/text-part order
after system-extension processing applies to all captured system text, including
earlier messages. Plain markers
and empty `user` values do not select a name. The attribute value is matched
literally, without XML entity decoding. Unsupported or malformed attributes
are left untouched and do not request a capture.

Configure `PROMPT_CAPTURE_SECRET` with the same value in Fly and Supabase Edge
Function secrets. The relay posts to `${SUPABASE_URL}/functions/v1/capture-prompt`
using `X-Prompt-Capture-Secret`. No caller Supabase session is required. Do not
log this secret or captured text. The helper in `prompt_capture.py` does not use
the user-JWT REST helpers in `supabase.py`.

Supabase accepts only one globally pending capture; an occupied slot skips the
new capture. The frontend admin page `/admin/prompt-captures` can copy the prompt,
save it permanently, or delete it. Save/delete release the slot. Saved prompts
remain accessible. Empty system text and capture bodies over 1 MiB are skipped.
Submission has a three-second deadline and never fails generation on storage
errors. A provider failure can still leave a captured request. There is no retry
queue, and requests without the marker make no capture network call.
