from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Define your configuration fields here with optional defaults
    environment: str = Field(..., env="ENVIRONMENT")
    app_name: str = Field(..., env="APP_NAME")
    debug: bool = Field(..., env="DEBUG")

    # database config
    # database_url: str = Field(..., env="DATABASE_URL")
    postgres_host: str = Field(..., env="POSTGRES_HOST")
    postgres_port: int = 5432  # default
    postgres_db: str = Field(..., env="POSTGRES_DB")
    postgres_user: str = Field(..., env="POSTGRES_USER")
    postgres_password: str = Field(..., env="POSTGRES_PASSWORD")

    # ingestion config
    arxiv_api_url: str = Field(..., env="ARXIV_API_URL")
    data_path: str = Field(..., env="DATA_PATH")

    # Generation model config
    temperature: float = Field(..., env="TEMPERATURE")
    top_p: float = Field(..., env="TOP_P")
    max_tokens: int = Field(..., env="MAX_TOKENS")

    # Comet config for Opik
    opik_api_key: str = Field(..., env="OPIK_API_KEY")
    opik_workspace: str = Field(..., env="OPIK_WORKSPACE")
    opik_project_name: str = Field(..., env="OPIK_PROJECT_NAME")

    # OpenAI config
    openai_model: str = Field(..., env="OPENAI_MODEL")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    rag_config: dict = {}

    class Config:
        env_file = ".env"  # Load variables from .env if they exist

settings = Settings()
