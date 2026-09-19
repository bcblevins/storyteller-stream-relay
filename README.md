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
