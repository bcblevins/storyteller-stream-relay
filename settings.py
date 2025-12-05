from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_JWT_SECRET: str
    SUPABASE_REST_URL: str
    SUPABASE_ANON_KEY: str  # Use anon key instead of service role

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
