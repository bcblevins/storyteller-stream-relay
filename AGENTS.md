# AGENTS.md

## Purpose (Automated Agents)
- This repo is the FastAPI streaming relay for Storyteller, which also includes a Vue 3 frontend and a Supabase backend (data/auth). Mention those only as context; all changes here should target the relay service.

## Scope and Boundaries
- Work only inside this repository.
- Do not modify or assume changes to the frontend or Supabase schema unless explicitly asked.

## Key Modules
- `app.py`: FastAPI app, CORS handling, SSE streaming, persistence, and reroll endpoints.
- `auth.py`: Supabase JWT verification (HS256 with `SUPABASE_JWT_SECRET`).
- `supabase.py`: Supabase REST helpers and data access for bots, conversations, messages, and message alternatives.
- `openai_service.py`: Async OpenAI-compatible streaming client wrapper.
- `settings.py`: Environment variable configuration via Pydantic settings.

## API Surface (Relay)
- `GET /healthz`: health check.
- `GET /auth/test`: validate auth + bot access.
- `GET /v1/message-by-stream-id`: resolve a persisted message by stream_id.
- `POST /v1/stream`: SSE streaming of model output; persists messages on completion.
- `POST /v1/reroll`: create a message alternative for reroll streaming.

## Data Flow Summary
- Stream requests resolve a bot (explicit bot_id -> conversation bot -> default bot), initialize the OpenAI-compatible client, stream tokens over SSE (`token`, `ping`, `done` events), then persist the final message to Supabase.
- Reroll requests create a `message_alternatives` record and return a stream_id for the client to stream into.

## Environment Variables
- Required: `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `SUPABASE_REST_URL`, `SUPABASE_ANON_KEY`.
- Optional CORS: `CORS_EXTRA_ORIGINS`, `CORS_ALLOW_ORIGIN_REGEX`.

## Behavior and Safety Notes
- Preserve SSE event names and payload shapes (`token`, `ping`, `done` with `stream_id` and optional IDs).
- Avoid logging secrets (API keys, raw JWTs).
- Keep persistence retry logic and id reconciliation intact.
