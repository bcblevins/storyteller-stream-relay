# Storyteller Relay

The **Storyteller Relay** is a specialized backend service designed to handle real-time AI streaming, authentication, and state persistence for the Storyteller application.

It serves as the secure "switchboard" between the client frontend, the database (Supabase), and Large Language Model providers (OpenAI/DeepSeek), ensuring that sensitive API keys remain server-side while delivering low-latency token streaming to users.

## 🎯 Why This Exists

Directly connecting a frontend to an LLM provider exposes API keys and lacks a robust way to persist generated content. This relay solves those problems by:

1.  **Securing Credentials:** It holds the LLM API keys, so the frontend never sees them.
2.  **Centralizing Auth:** It verifies Supabase JWTs before processing any request.
3.  **Persisting State:** It automatically saves AI-generated messages to the database immediately after streaming completes, ensuring the conversation history is always in sync.
4.  **Handling "Rerolls":** It manages the logic for regenerating (rerolling) specific message nodes in the story tree without breaking the narrative flow.

## 🏗️ Architecture

The service is built with **FastAPI** and designed to be stateless and scalable.

### Core Flows

* **Streaming Pipeline:**
    * **Input:** Receives a user message context + system prompt + Supabase Auth Token.
    * **Process:** Authenticates the user, fetches the appropriate "Bot" configuration (model, temperature, system instructions) from Supabase, and opens a stream to the LLM provider.
    * **Output:** Streams tokens back to the client via Server-Sent Events (SSE).
    * **Cleanup:** On stream completion (or interruption), it asynchronously saves the full message to Supabase.

* **Bot Resolution Strategy:**
    The relay dynamically decides which AI persona to use for a response based on a priority hierarchy:
    1.  **Explicit:** A specific `bot_id` passed in the payload.
    2.  **Conversation-Bound:** The bot assigned to the specific conversation ID.
    3.  **Default:** The user's preferred default bot.
    4.  **Fallback:** The most recently updated bot owned by the user.

### Key Features

* **Server-Sent Events (SSE):** Uses `sse_starlette` for efficient, real-time text streaming.
* **Smart Persistence:** Features retry logic ("safe post") to ensure messages are saved to the database even if the network hiccups.
* **Rate Limiting:** Includes basic in-memory rate limiting to prevent abuse.
* **CORS Management:** specialized handling to support secure cross-origin requests from the Storyteller frontend.

## Creator Session Native Tools

Creator sessions now support a relay-managed native tool loop with real upstream `tools` usage.

* `POST /v1/creator/stream`
  * Default behavior stays the same for plain creator text streaming.
  * When `mode` is `"native_tools"`, the relay accepts `messages` whose `content` is either a plain string or an ordered array of native content blocks such as `text`, `tool_use`, and `tool_result`.
  * The relay forwards that creator message history upstream as the prompt source of truth, apart from normal provider-format translation, and does not reconstruct tool turns.
* `POST /v1/creator/stream/continue`
  * Continues the same creator native-tool turn using the provided `messages` history as-is.
  * Legacy fields like `decision`, `tool_call`, `tool_result`, and `feedback` may still be sent for telemetry, but they no longer modify prompt history.

### Native-tool SSE events

* `creator_tool_call_start`
  * JSON payload with nullable `tool_name`, emitted as soon as upstream tool-call generation begins.
* `creator_tool_call`
  * JSON payload with `stream_id`, `status`, `mode`, a normalized `tool_call` object (`id`, `name`, `arguments`, `raw_arguments`), plus legacy flat fields (`tool_call_id`, `tool_name`, `arguments`, `raw_arguments`) for compatibility, optional `assistant_content`, `finish_reason`, and optional `usage`.
* `token`
  * Final assistant summary text after the tool loop is complete, or direct assistant text if no tool call is needed.
* `done`
  * JSON payload with `stream_id`, `status` (`awaiting_tool_approval` or `completed`), `mode`, `finish_reason`, and optional `usage`.
* `error`
  * JSON payload with clean upstream/provider failure details. Tool capability gating is not done in the relay for v1.

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

### Reasoning Injection

The relay enables provider-specific thinking/reasoning defaults server-side and preserves explicit request settings unless override is enabled.

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

When `ENABLE_SYSTEM_INJECTION_TAG=true`, the proxy scans system messages for
`<injection>...</injection>` (or your configured tag), removes those blocks from
system content, and appends the extracted text to the latest message before
sending to OpenRouter.

When `ENABLE_SYSTEM_THINKING_TAG=true`, the proxy scans system messages for
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
