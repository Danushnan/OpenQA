from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MISTRAL_API_KEY: str = Field(...)

    model_config = SettingsConfigDict(env_file=".env")
    
settings = Settings()