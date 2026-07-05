# Contract — Relay Persistence Decoupling (Change A)

**Status:** proposed, not yet implemented
**Owners:** frontend (storyteller/app) + relay (storyteller-stream-relay), implemented independently against this document
**Purpose of this file:** a shared, checkable interface contract so the two repos can change in step without coordinating line-by-line. Each side implements to this doc; both run the Verification Checklist at the end after implementation.

---

## 1. Conceptual goal

The relay is a **relay**. It should stream model tokens and nothing else. All persistence of chat messages moves to the frontend, which already writes these tables directly under the same user JWT.

> **The relay must never read or write `messages` or `message_alternatives` (or any conversation-message state) in Supabase.** It authenticates the request, calls the model, streams tokens back, and terminates. The frontend owns the entire message lifecycle: creation, ids, streaming/complete flags, alternatives, and cleanup.

This is safe because the frontend already inserts messages, alternatives, and director notes directly into Supabase with the same JWT the relay currently borrows — the relay is not a trust boundary, only an indirection. Removing that indirection deletes the temp-id reconciliation handshake and means **the database only ever holds completed messages** (streaming becomes a purely in-memory, client-side phase).

Accepted tradeoff (product decision, already made): if the client disappears mid-generation — or the tab is closed before a generation completes — that reply is **lost**, not recovered. The "resend / regenerate" affordance covers this. The relay therefore has **no durability obligation** once the connection drops.

---

## 2. The contract in one line

**Given a generation request carrying fully-assembled context, the relay streams the model's output token-by-token and terminates with exactly one terminal event, touching no conversation-message rows in the database.**

Everything else is the frontend's responsibility.

---

## 3. Current state (what the relay does today — for the relay agent to remove)

These are the behaviors being deleted. Line refs are into the relay repo (`storyteller-stream-relay`) unless noted.

- **Fresh reply persistence:** after `/v1/stream` completes, inserts a row into `messages` via `post_message` (`app.py:692`, `app.py:704`, `supabase.py:227`).
- **Reroll placeholder + patch:** `/v1/reroll` verifies the parent from `messages`, resolves temp parent ids via `messages.stream_id`, inserts an empty `message_alternatives` placeholder, then patches it on completion (`app.py:997`–`app.py:1069`, `supabase.py:242`, `supabase.py:246`).
- **Disconnect/error persistence:** on client disconnect (`aborted=true`) or upstream error, still writes the partial buffer to the DB with completion flags (`app.py:631`, `app.py:659`, `app.py:697`; error path persists partial as `is_complete=true` — a latent "truncated-but-looks-complete" bug that disappears under this change).
- **DB reads to build context / resolve ids:** `/v1/reroll` and `/v1/message-by-stream-id` read `messages` (`app.py:829`, `supabase.py:298`, `supabase.py:312`).

Frontend side that will change in lockstep (for reference; frontend agent owns these):
- Stops reading a DB id from the `done` event; assigns the id from its own insert (removes `reconcileMessageId`, `MessageOrchestrator.ts:486`).
- Sends fully-assembled context for **reroll** the same way it does for a normal stream (today reroll sends only `{parent_message_id, conversation_id}` and relies on the relay to rebuild context — `relay.ts:104`).
- Treats `error` as terminal (today it waits for a `done` after `error` — `relay.ts:331`).

---

## 4. Target relay behavior

### MUST
1. Authenticate the request via the `Authorization: Bearer <supabase jwt>` header (for identity / rate-limiting only).
2. Build the model prompt **solely from the `messages` in the request body**. No Supabase read to assemble context.
3. Stream model output as SSE `token` events (see §5).
4. Terminate every stream with **exactly one** terminal event: `done` (success) **xor** `error` (failure). Never emit content after a terminal event.
5. On client disconnect: stop generating and release resources. Write nothing. Emit nothing (the client already knows).

### MUST NOT
6. Read or write `messages` or `message_alternatives`, or any conversation-message row, for any streaming path (fresh or reroll).
7. Depend on `parent_message_id`, `alternative_id`, or `stream_id` for persistence. The relay does not create, own, or return message/alternative database ids.
8. Persist partial output on disconnect or error.

### MAY
9. Remove `/v1/reroll` and `/v1/message-by-stream-id` entirely once the frontend stops calling them (see §6). A reroll is just a `/v1/stream` call with reroll context.
10. Include optional, advisory generation metadata in the `done` payload (e.g. finish reason). Not required, and never persisted by the relay.

### OUT OF SCOPE (leave alone)
- `/v1/openrouter/demo` bot provisioning (`app.py` demo bot path) — unrelated to message persistence; do **not** remove.
- Any auth/rate-limit/logging that does not touch conversation-message rows.

---

## 5. Target wire protocol

### Request (frontend → relay)
```
POST /v1/stream
Authorization: Bearer <supabase jwt>
Content-Type: application/json

{
  "messages": [ ...fully-assembled context... ],   // relay does NOT read the DB to build this
  "conversation_id": <id>,                          // logging / rate-limit / routing only; not for persistence
  "bot_id": <id>,                                   // model/bot selection (optional)
  "max_tokens": <n>                                 // optional
  // additional model params (reasoning toggles, etc.) as today
}
```
- The **same shape is used for rerolls.** The frontend assembles the regeneration context and sends it here. `is_alternative`, `alternative_id`, and `stream_id` are no longer sent and can be dropped by the relay.

### Response — SSE events (relay → frontend)
| Event | Data | Meaning | Terminal? |
|---|---|---|---|
| `token` | text delta | primary content chunk | no |
| `reasoning` | text delta | ephemeral "thinking" (not persisted by anyone) | no |
| `ping` | — | keep-alive, ignored by client | no |
| `done` | optional JSON metadata | generation finished successfully | **yes** |
| `error` | JSON `{ "error": "<message>" }` | generation failed | **yes** |

- The frontend already understands `token` / `reasoning` / `ping` / `done` / `error` (`relay.ts:314`–`360`). Keeping this vocabulary minimizes frontend churn.
- **`done` payload:** the client will ignore any `message_id` / `alternative_id`. The relay is not required to send them. Any generation metadata the relay includes (e.g. finish reason) is optional and advisory only.
- **Terminal guarantee (critical):** the client unblocks on the first terminal event. The relay must send exactly one and never emit `token` after it. In particular, `error` is terminal — do **not** follow it with `done`.

---

## 6. Outcome behavior — target ("in effect")

| Outcome | Relay does | Frontend does |
|---|---|---|
| **Normal completion** | streams all `token`s, emits `done`. No DB write. | on `done`, inserts the completed message row and adopts the id from its own insert. |
| **Reroll completion** | identical to normal completion (same endpoint, reroll context). | on `done`, inserts the completed variant row. |
| **Upstream/model error** | emits `error` (terminal). No DB write. | on `error` (terminal), surfaces the failure and discards the in-memory partial per its error policy. |
| **User cancel** | sees a dropped connection; stops; writes nothing. | client initiated the abort and holds the partial in memory. If "keep on cancel" is desired, the client persists the partial itself — **relay-independent** (frontend decision, not part of this contract). |
| **Passive disconnect** | stops; writes nothing. | reply is lost; user regenerates. |

Note the cancel vs. disconnect distinction that used to be impossible relay-side (both are just a dropped connection) is now trivially available **client-side**, because the client knows which one it triggered and already has the streamed text. That is a frontend concern and deliberately not in this contract.

---

## 7. Verification checklist (run by BOTH sides after implementation)

**Relay:**
- [ ] `grep` of the relay for `messages` / `message_alternatives` shows no read or write anywhere on the conversation-streaming paths.
- [ ] `/v1/stream` builds context only from the request body — no Supabase message read.
- [ ] Rerolls need no DB round-trip; `/v1/reroll` and `/v1/message-by-stream-id` are removed or provably unused.
- [ ] Every stream ends in exactly one terminal event; no `token` after a terminal; `error` is not followed by `done`.
- [ ] Client disconnect writes nothing.
- [ ] `/v1/openrouter/demo` still present and working.

**Frontend:**
- [ ] AI reply and reroll variant are persisted by the client on `done` (insert-on-completion), id taken from the client's own insert.
- [ ] No code reads `message_id` / `alternative_id` from the `done` payload; `reconcileMessageId` against relay ids removed.
- [ ] Reroll sends fully-assembled context to the streaming endpoint (no reliance on relay-side context rebuild).
- [ ] `error` is treated as terminal (no waiting for a trailing `done`).
- [ ] No dependence on incomplete/streaming rows existing in the DB (incomplete-stream recovery via `rpc_complete_stream` can be retired).

---

## 8. Sequencing

This is **Change A**. It is a prerequisite that de-risks **Change B** (collapsing `messages` + `message_alternatives` into one self-referential table): once the relay and the `rpc_*_stream` RPCs no longer write messages, Change B shrinks to a client-adapter + schema change with a single writer. Do A first, verify via this checklist, then scope B.
