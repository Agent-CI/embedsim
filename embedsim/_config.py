from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedSimSettings(BaseSettings):
    """Configuration settings for embedsim.

    Settings are loaded from environment variables with the EMBEDSIM_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDSIM_",
        case_sensitive=False,
    )

    # Default model to use
    model: str = "openai/text-embedding-3-small"

    # OpenAI API key (can also use OPENAI_API_KEY)
    openai_api_key: str | None = None

    def get_openai_api_key(self) -> str | None:
        """Get OpenAI API key, checking both EMBEDSIM_OPENAI_API_KEY and OPENAI_API_KEY."""
        import os

        return self.openai_api_key or os.getenv("OPENAI_API_KEY")
