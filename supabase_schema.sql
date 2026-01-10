-- THIS IS NOT A COMPLETE SCHEMA. ONLY TABLES RELEVANT TO THE CURRENT CONTEXT ARE INCLUDED.
-- RLS POLICIES, INDEXES, AND NON-RELAY TABLES ARE INTENTIONALLY OMITTED FOR A PUBLIC REPO.

CREATE TABLE public.bots (
  id integer NOT NULL DEFAULT nextval('bots_id_seq'::regclass),
  model text NOT NULL,
  access_key text,
  access_path text,
  context_size integer DEFAULT 10000,
  max_tokens integer DEFAULT 1000,
  advanced_prompt text,
  temperature double precision DEFAULT 0.7,
  name text,
  is_default boolean NOT NULL DEFAULT false CHECK (is_default = ANY (ARRAY[true, false])),
  user_id uuid,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  reasoning boolean,
  show_reasoning boolean,
  is_openrouter boolean NOT NULL DEFAULT false,
  openrouter_key text,
  CONSTRAINT bots_pkey PRIMARY KEY (id),
  CONSTRAINT bots_user_uuid_fkey FOREIGN KEY (user_id) REFERENCES public.users(id),
  CONSTRAINT bots_openrouter_key_fkey FOREIGN KEY (openrouter_key) REFERENCES public.or_keys(or_key)
);
CREATE TABLE public.conversations (
  id integer NOT NULL DEFAULT nextval('conversations_id_seq'::regclass),
  character_id integer NOT NULL,
  persona_id integer NOT NULL,
  title text NOT NULL DEFAULT 'Conversation'::text,
  created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
  user_id uuid,
  message_count integer NOT NULL DEFAULT 0,
  last_message_at timestamp with time zone,
  bot_id integer,
  CONSTRAINT conversations_pkey PRIMARY KEY (id),
  CONSTRAINT conversations_character_id_fkey FOREIGN KEY (character_id) REFERENCES public.characters(id),
  CONSTRAINT conversations_persona_id_fkey FOREIGN KEY (persona_id) REFERENCES public.personas(id),
  CONSTRAINT conversations_user_uuid_fkey FOREIGN KEY (user_id) REFERENCES public.users(id),
  CONSTRAINT conversations_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bots(id)
);
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
CREATE TABLE public.messages (
  id integer NOT NULL DEFAULT nextval('messages_id_seq'::regclass),
  conversation_id integer NOT NULL,
  content text NOT NULL,
  t timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
  is_user_author boolean NOT NULL,
  is_active boolean DEFAULT true,
  is_streaming boolean DEFAULT false,
  is_complete boolean DEFAULT true,
  stream_id text,
  user_id uuid,
  CONSTRAINT messages_pkey PRIMARY KEY (id),
  CONSTRAINT messages_conversation_id_fkey FOREIGN KEY (conversation_id) REFERENCES public.conversations(id),
  CONSTRAINT messages_user_uuid_fkey FOREIGN KEY (user_id) REFERENCES public.users(id)
);
CREATE TABLE public.or_keys (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  user_id uuid NOT NULL DEFAULT auth.uid(),
  or_key text NOT NULL UNIQUE,
  CONSTRAINT or_keys_pkey PRIMARY KEY (id)
);

CREATE OR REPLACE FUNCTION public.create_demo_openrouter_bot(
  p_user_id uuid,
  p_or_key text,
  p_model text,
  p_access_path text,
  p_name text
)
RETURNS TABLE (bot_id integer)
LANGUAGE plpgsql
AS $$
DECLARE
  v_is_default boolean;
  v_bot_id integer;
BEGIN
  SELECT NOT EXISTS(
    SELECT 1 FROM public.bots WHERE user_id = p_user_id AND is_default = true
  ) INTO v_is_default;

  INSERT INTO public.or_keys (user_id, or_key)
  VALUES (p_user_id, p_or_key);

  INSERT INTO public.bots (
    model,
    access_key,
    access_path,
    name,
    temperature,
    is_default,
    user_id,
    is_openrouter,
    openrouter_key
  )
  VALUES (
    p_model,
    NULL,
    p_access_path,
    p_name,
    0.3,
    v_is_default,
    p_user_id,
    true,
    p_or_key
  )
  RETURNING id INTO v_bot_id;

  RETURN QUERY SELECT v_bot_id;
END;
$$;
