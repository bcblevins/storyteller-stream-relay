# AGENTS.md

## Purpose (Automated Agents)
- This repo is the FastAPI streaming relay for Storyteller, which also includes a Vue 3 frontend and a Supabase backend (data/auth). Mention those only as context; all changes here should target the relay service.

## Scope and Boundaries
- Work only inside this repository.
- Do not modify or assume changes to the frontend or Supabase schema unless explicitly asked.

## Key Modules
- `app.py`: FastAPI app, CORS handling, SSE streaming, and OpenRouter demo provisioning.
- `auth.py`: Supabase JWT verification (HS256 with `SUPABASE_JWT_SECRET`).
- `supabase.py`: Supabase REST helpers for resolving bots (including workspace-conversation bots) and OpenRouter demo bots. Read-only as far as application data is concerned.
- `openai_service.py`: Async OpenAI-compatible streaming client wrapper.
- `settings.py`: Environment variable configuration via Pydantic settings.

## API Surface (Relay)
- `GET /healthz`: health check.
- `GET /auth/test`: validate auth + bot access.
- `POST /v1/chat/completions`: **external-consumer** OpenAI-compatible passthrough (proxy API key auth, not Storyteller JWT). Not part of the Storyteller app surface; keep its contract stable and exclude it from Storyteller refactors.
- `POST /v1/stream`: the only generation endpoint. SSE streaming of model output; persists nothing. A request carrying `tools` is served by the tool-aware path, which emits `tool_call_start` / `tool_call` alongside `token`.
- `POST /v1/openrouter/demo`: provision an OpenRouter demo bot.

## Data Flow Summary
- Stream requests resolve a bot (explicit bot_id -> conversation bot -> default bot), initialize the OpenAI-compatible client, stream tokens over SSE (`token`, `reasoning`, `ping`, `done` or `error` events), and write nothing. Tools are a capability of a request, not of a surface: any stream may carry them.
- Rerolls, Creator turns, and tool continuations are all normal `/v1/stream` requests with frontend-assembled context. The relay stores nothing for any of them; the app owns all persistence.

## Environment Variables
- Required: `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `SUPABASE_REST_URL`, `SUPABASE_ANON_KEY`.
- Optional CORS: `CORS_EXTRA_ORIGINS`, `CORS_ALLOW_ORIGIN_REGEX`.

## Behavior and Safety Notes
- Preserve SSE event names and payload shapes (`token`, `reasoning`, `ping`, `done`, `error`).
- Emit exactly one terminal event: `done` on success or `error` on failure. On client disconnect, stop and emit nothing.
- The relay does not write application data. It has no message-persistence path at all — do not reintroduce one.
- Avoid logging secrets (API keys, raw JWTs).
