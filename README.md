
# Storyteller Streaming Relay

A lightweight, secure **FastAPI + SSE** microservice that streams tokens from any OpenAI-compatible model (e.g. DeepSeek, OpenAI, local LLMs) to the Storyteller web app in real time.

This relay acts as a **secure middle layer** between authenticated users and language-model providers.  
It keeps API keys off the client, validates users via Supabase Auth JWTs, and persists completed messages back into Supabase.

---

## 🌐 Overview

### Architecture
```

Client (Storytellr.me)
│  (Supabase JWT)
▼
FastAPI Relay  ──→  OpenAI-compatible API (stream)
│
└──→  Supabase REST (messages, bots)

````

### Key Features
- **Supabase-based AuthN/AuthZ**  
  Uses Supabase JWT verification against the project JWKS.
- **Secure bot key retrieval**  
  Relay fetches provider credentials using Supabase’s service role key (never exposed client-side).
- **Async SSE token streaming**  
  Real-time token emission via `EventSourceResponse`.
- **Automatic persistence**  
  On completion or disconnect, the final message is written back to Supabase.
- **Graceful handling**  
  Detects disconnects, timeouts, and emits structured `error`, `ping`, and `done` events.
- **Portable deployment**  
  Run locally behind Cloudflare Tunnel or move unchanged to a VPS or container.

---

## ⚙️ Requirements

- Python 3.10+
- Installed dependencies:
```bash
  pip install fastapi uvicorn sse-starlette httpx python-jose[cryptography] \
      pydantic-settings openai orjson
```

* Supabase project with:

  * `bots` table (includes `access_key`, `access_path`, `model`, etc.)
  * `messages` table (with a unique index on `stream_id`)

---

## 🧩 Environment Variables

Create a `.env` file in the project root:

```bash
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_JWKS_URL=https://<project>.supabase.co/auth/v1/keys
SUPABASE_REST_URL=https://<project>.supabase.co/rest/v1
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
```

---

## 🏗️ Project Structure

```
storytellr-relay/
├─ app.py               # FastAPI app with /v1/stream endpoint
├─ auth.py              # Supabase JWT verification
├─ supabase.py          # fetch_bot(), post_message(), safe_post()
├─ openai_service.py    # AsyncOpenAI wrapper for streaming
├─ settings.py          # env configuration (pydantic-settings)
├─ .env
└─ README.md
```

---

## 🚀 Running Locally

1. Start the relay:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Verify health:

   ```bash
   curl http://localhost:8000/healthz
   # → {"status": "ok"}
   ```

3. Expose publicly through Cloudflare Tunnel:

   ```bash
   cloudflared tunnel run storytellr-tunnel
   ```

   Add ingress mapping in `~/.cloudflared/config.yml`:

   ```yaml
   - hostname: api.storytellr.me
     service: http://localhost:8000
   ```

---

## 🧠 API Reference

### `GET /healthz`

Health check → `{ "status": "ok" }`

### `POST /v1/stream`

Streams assistant tokens via Server-Sent Events (SSE).

**Headers**

```
Authorization: Bearer <Supabase JWT>
Content-Type: application/json
```

**Body**

```json
{
  "bot_id": 123,
  "conversation_id": 456,
  "prompt": "Tell me a story about dragons.",
  "system": "optional system prompt",
  "stream_id": "optional-client-uuid"
}
```

**SSE events**

| Event   | Data                      |
| ------- | ------------------------- |
| `token` | partial content token     |
| `ping`  | keepalive heartbeat       |
| `error` | structured JSON error     |
| `done`  | `{ "stream_id": "<id>" }` |

On stream completion, the final joined message is persisted to Supabase.

---

## 🧱 Data Persistence

The relay writes each generated message to Supabase:

| Field             | Description                  |
| ----------------- | ---------------------------- |
| `user_id`         | from JWT `sub`               |
| `conversation_id` | client-provided              |
| `content`         | concatenated output          |
| `is_user_author`  | always `false`               |
| `is_streaming`    | always `false`               |
| `is_complete`     | false if client disconnected |
| `stream_id`       | unique per generation        |

Create the index once:

```sql
create unique index if not exists messages_stream_id_unique on messages(stream_id);
```

---

## 🧰 Optional Advanced Features (Phase 5)

* `/v1/streams/{stream_id}/cancel` → cancel active stream
* Timeout protection (default 120 s)
* In-memory per-user rate limit (5/min)
* Structured `error` SSE events
* Periodic `ping` to keep Cloudflare Tunnel alive

---

## 🧾 Logging

JSON-style structured logs include:

```
timestamp  level  user_id  bot_id  conversation_id  stream_id  latency_ms
```

Example:

```
2025-10-25 14:32:10 INFO Stream complete (s-1735182730) - 584 tokens
2025-10-25 14:32:10 INFO Message persisted for stream s-1735182730 (complete=True)
```

---

## 🧱 Future Work

* Redis-backed rate limiting & cancel flags
* Metrics endpoint (`/metrics`) for observability
* Containerized deployment (`Dockerfile`)
* Multi-provider abstraction layer

---

## 📜 License

MIT © 2025 Storyteller
This project is part of the **Storyteller** ecosystem for interactive storytelling.

