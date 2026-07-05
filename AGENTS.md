# AGENTS.md

## Purpose (Automated Agents)
- This repo is the FastAPI streaming relay for Storyteller, which also includes a Vue 3 frontend and a Supabase backend (data/auth). Mention those only as context; all changes here should target the relay service.

## Scope and Boundaries
- Work only inside this repository.
- Do not modify or assume changes to the frontend or Supabase schema unless explicitly asked.

## Key Modules
- `app.py`: FastAPI app, CORS handling, SSE streaming, creator persistence, and OpenRouter demo provisioning.
- `auth.py`: Supabase JWT verification (HS256 with `SUPABASE_JWT_SECRET`).
- `supabase.py`: Supabase REST helpers and data access for bots, creator sessions, creator messages, workspace conversations, and OpenRouter demo bots.
- `openai_service.py`: Async OpenAI-compatible streaming client wrapper.
- `settings.py`: Environment variable configuration via Pydantic settings.

## API Surface (Relay)
- `GET /healthz`: health check.
- `GET /auth/test`: validate auth + bot access.
- `POST /v1/stream`: SSE streaming of model output for conversation messages; no conversation-message persistence.
- `POST /v1/creator/stream`: creator SSE streaming, including native tool mode.
- `POST /v1/creator/stream/continue`: continue a creator native-tool turn.
- `POST /v1/openrouter/demo`: provision an OpenRouter demo bot.

## Data Flow Summary
- Conversation stream requests resolve a bot (explicit bot_id -> conversation bot -> default bot), initialize the OpenAI-compatible client, stream tokens over SSE (`token`, `reasoning`, `ping`, `done` or `error` events), and write nothing to conversation-message tables.
- Rerolls are normal `/v1/stream` requests with frontend-assembled context. The frontend owns message, alternative, and cleanup persistence.
- Creator stream paths may persist to creator-specific tables (`creator_messages` / `conversation_messages`) and are separate from conversation-message persistence.

## Environment Variables
- Required: `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `SUPABASE_REST_URL`, `SUPABASE_ANON_KEY`.
- Optional CORS: `CORS_EXTRA_ORIGINS`, `CORS_ALLOW_ORIGIN_REGEX`.

## Behavior and Safety Notes
- Preserve SSE event names and payload shapes (`token`, `reasoning`, `ping`, `done`, `error`).
- For conversation `/v1/stream`, emit exactly one terminal event: `done` on success or `error` on failure. On client disconnect, stop and write/emit nothing.
- Do not read or write `messages` or `message_alternatives` from relay conversation streaming paths.
- Avoid logging secrets (API keys, raw JWTs).
