from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    AWS_REGION: str = "us-west-2"

    BEDROCK_AWS_ACCESS_KEY_ID: str | None = None
    BEDROCK_AWS_SECRET_ACCESS_KEY: str | None = None

    bedrock_default_model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"


settings = Settings()
