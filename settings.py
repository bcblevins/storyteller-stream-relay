from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_JWT_SECRET: str
    SUPABASE_REST_URL: str
    SUPABASE_ANON_KEY: str  # Use anon key instead of service role
    OPENROUTER_PROVISIONING_KEY: str
    OPENROUTER_DEMO_MODEL: str
    OPENROUTER_DEMO_LIMIT: float
    OPENROUTER_DEMO_LIMIT_RESET: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
