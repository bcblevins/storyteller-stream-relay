from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_JWT_SECRET: str
    SUPABASE_REST_URL: str
    SUPABASE_ANON_KEY: str  # Use anon key instead of service role
    OPENROUTER_PROVISIONING_KEY: str
    OPENROUTER_DEMO_MODEL: str
    OPENROUTER_DEMO_LIMIT: float
    OPENROUTER_DEMO_LIMIT_RESET: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Optional GLM passthrough feature flags (safe defaults keep existing app behavior unchanged)
    GLM_PROXY_API_KEY: Optional[str] = None
    FORCE_REASONING_ENABLED: bool = False
    FORCE_REASONING_EFFORT: str = "high"
    FORCE_REASONING_MODEL_PATTERNS: str = "z-ai/glm-4.6:nitro"
    FORCE_REASONING_OVERRIDE: bool = False
    ENABLE_SYSTEM_INJECTION_TAG: bool = True
    SYSTEM_INJECTION_TAG_NAME: str = "injection"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def force_reasoning_model_patterns_list(self) -> tuple[str, ...]:
        patterns = [p.strip() for p in self.FORCE_REASONING_MODEL_PATTERNS.split(",") if p.strip()]
        return tuple(patterns)

settings = Settings()
