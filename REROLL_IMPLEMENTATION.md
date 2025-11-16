# Reroll Implementation for Relay Server

## Overview

The relay server now supports reroll functionality through two new endpoints that work together to create and stream alternative messages.

## Endpoints

### 1. POST `/v1/reroll`

Creates a new alternative message and returns the details needed for streaming.

**Request Body:**
```json
{
  "parent_message_id": 123,
  "conversation_id": 456
}
```

**Response:**
```json
{
  "alternative_message": {
    "id": 789,
    "parent_message_id": 123,
    "conversation_id": 456,
    "content": "",
    "is_user_author": false,
    "is_streaming": true,
    "is_complete": false,
    "stream_id": "reroll-1700134567890",
    "is_active": true
  },
  "stream_id": "reroll-1700134567890"
}
```

### 2. Enhanced POST `/v1/stream`

Now supports alternative messages with additional parameters:

**New Request Body Fields:**
```json
{
  "is_alternative": true,
  "alternative_id": 789,
  "messages": [...],
  "conversation_id": 456,
  "stream_id": "reroll-1700134567890"
}
```

## Frontend Integration Flow

### Step 1: Create Alternative
1. Call `/v1/reroll` with `parent_message_id` and `conversation_id`
2. Store the returned `alternative_message` in your local state
3. Use the `alternative_message.id` and `stream_id` for streaming

### Step 2: Stream Alternative Content
1. Call `/v1/stream` with:
   - `is_alternative: true`
   - `alternative_id: <alternative_message.id>`
   - `stream_id: <stream_id_from_reroll>`
   - `messages` array (same as normal streaming)
   - `conversation_id`

### Step 3: Handle Completion
- The alternative message will be automatically updated in Supabase
- Frontend can update local state with the completed content

## Database Schema

The implementation uses the existing `message_alternatives` table:

```sql
CREATE TABLE public.message_alternatives (
  id integer NOT NULL DEFAULT nextval('message_alternatives_id_seq'::regclass),
  conversation_id integer NOT NULL,
  parent_message_id integer NOT NULL,
  content text NOT NULL,
  t timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
  is_user_author boolean NOT NULL,
  is_active boolean DEFAULT true,
  is_streaming boolean DEFAULT false,
  is_complete boolean DEFAULT true,
  stream_id text,
  user_id uuid,
  CONSTRAINT message_alternatives_pkey PRIMARY KEY (id),
  CONSTRAINT message_alternatives_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES public.conversations(id),
  CONSTRAINT message_alternatives_parent_message_id_fkey FOREIGN KEY (parent_message_id) REFERENCES public.messages(id),
  CONSTRAINT message_alternatives_user_uuid_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
```

## Error Handling

- **400 Bad Request**: Missing required fields
- **404 Not Found**: Parent message not found or unauthorized
- **400 Bad Request**: Attempting to reroll user messages
- **500 Internal Server Error**: Database or AI service failures

## Key Features

1. **Seamless Integration**: Uses same streaming infrastructure as regular messages
2. **Batch Persistence**: Alternative messages are updated after streaming completes
3. **Error Recovery**: Retry logic for database operations
4. **Authentication**: JWT-based user verification for all operations
5. **Stream Management**: Unique stream IDs for tracking and recovery

## Example Frontend Usage

```javascript
// Step 1: Create alternative
const rerollResponse = await apiService.rerollMessage(parentMessageId, conversationId);
const alternative = rerollResponse.alternative_message;

// Step 2: Stream alternative content
await apiService.generateTextStream(
  messages,
  onChunk,
  onError,
  onComplete,
  onConnected,
  abortSignal,
  conversationId,
  {
    is_alternative: true,
    alternative_id: alternative.id,
    stream_id: rerollResponse.stream_id
  }
);
```

## Notes

- Only AI messages can be rerolled (user messages will return 400)
- Parent message ownership is verified for security
- Alternative messages maintain the same conversation context
- The system maintains compatibility with existing frontend code
