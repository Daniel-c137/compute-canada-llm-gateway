from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)

    upstream_base_url: str = "http://127.0.0.1:8000"
    gateway_token: str | None = None
    protect_docs: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
