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
- `continuation_stream.py`: Opt-in assistant-envelope assembly for client-owned tool loops; preserves provider continuation fields and validates completion.
- `settings.py`: Environment variable configuration via Pydantic settings.

## API Surface (Relay)
- `GET /healthz`: health check.
- `GET /auth/test`: validate auth + bot access.
- `POST /v1/chat/completions`: **external-consumer** OpenAI-compatible passthrough (proxy API key auth, not Storyteller JWT). Not part of the Storyteller app surface; keep its contract stable and exclude it from Storyteller refactors. `<capture-content/>` in any message text opts into best-effort system prompt capture through Supabase; the marker is stripped after system extension and before injection/thinking transforms. Optional `user="Name"` replaces literal name occurrences with `{{user}}` only in captured system text; the first nonempty value wins. The first complete `<system-extension>...</system-extension>` block is moved to the end of the last system message (or a new prepended system message) before capture, injection and thinking transforms.
- `POST /mock/v1/chat/completions`: external-consumer OpenAI-compatible mock using the same proxy API key. Returns fixed assistant text immediately, with SSE when `stream: true` and ordinary completion JSON otherwise. No provider calls, transforms, prompt capture, or persistence.
- `POST /v1/stream`: the only generation endpoint. SSE streaming of model output; persists nothing. A request carrying `tools` is served by the tool-aware path, which emits `tool_call_start` / `tool_call` alongside `token`.
  - `tool_protocol: "continuation_v1"` opts into the terminal assistant envelope described in `docs/continuation-protocol.md`, for tool or text-only rounds. Requests without it retain the legacy payload shapes and one-call behavior.
  - Optional `response_format` requests JSON mode or JSON Schema output; `request_transforms.py` translates provider capabilities. The app validates its domain output schema.
  - Optional `temperature` overrides the selected bot for this request. It must be a finite number from 0 to 2; zero is preserved.
- `POST /v1/openrouter/demo`: provision an OpenRouter demo bot.

## Data Flow Summary
- Stream requests resolve a bot (explicit bot_id -> conversation bot -> default bot), initialize the OpenAI-compatible client, stream tokens over SSE (`token`, `reasoning`, `ping`, `done` or `error` events), and write nothing. Tools are a capability of a request, not of a surface: any stream may carry them.
- Rerolls, Creator turns, and tool continuations are all normal `/v1/stream` requests with frontend-assembled context. The relay stores nothing for any of them; the app owns all persistence.

## Environment Variables
- Required: `SUPABASE_URL`, `SUPABASE_JWT_SECRET`, `SUPABASE_REST_URL`, `SUPABASE_ANON_KEY`.
- Optional external diagnostics: `PROMPT_CAPTURE_SECRET` authenticates submissions to the Supabase `capture-prompt` Edge Function.
- Optional CORS: `CORS_EXTRA_ORIGINS`, `CORS_ALLOW_ORIGIN_REGEX`.

## Behavior and Safety Notes
- Preserve SSE event names and payload shapes (`token`, `reasoning`, `ping`, `done`, `error`).
- Emit exactly one terminal event: `done` on success or `error` on failure. On client disconnect, stop and emit nothing.
- Application generation endpoints have no message-persistence path. The separate external endpoint can submit opt-in system prompt diagnostics to Supabase; capture retention and admin operations belong to Supabase and the frontend.
- Avoid logging secrets (API keys, raw JWTs).
