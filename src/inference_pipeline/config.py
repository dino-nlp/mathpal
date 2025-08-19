from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # Embeddings config
    EMBEDDING_MODEL_ID: str = "BAAI/bge-m3"
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = 512
    EMBEDDING_SIZE: int = 1024  # Updated to match BAAI/bge-m3 actual dimension
    EMBEDDING_MODEL_DEVICE: str = "cpu"

    # OpenRouter config
    OPENROUTER_KEY: str | None = None
    OPENROUTER_BASE_MODEL: str = "openrouter/openai/gpt-oss-20b:free"

    # QdrantDB config
    QDRANT_DATABASE_HOST: str = "localhost"  # Or 'qdrant' if running inside Docker
    QDRANT_DATABASE_PORT: int = 6333

    USE_QDRANT_CLOUD: bool = (
        False  # if True, fill in QDRANT_CLOUD_URL and QDRANT_APIKEY
    )
    QDRANT_CLOUD_URL: str = "str"
    QDRANT_APIKEY: str | None = None


    # CometML config
    COMET_API_KEY: str | None = None
    COMET_WORKSPACE: str | None = None
    COMET_PROJECT: str | None = None

    # LLM Model config
    HUGGINGFACE_ACCESS_TOKEN: str | None = None
    MODEL_ID: str = "unsloth/gemma-3n-E2B-it" # Change this with your Hugging Face model ID to test out your fine-tuned LLM

    MAX_INPUT_TOKENS: int = 1024  # Max length of input text.
    MAX_TOTAL_TOKENS: int = 2048  # Max length of the generation (including input text).
    MAX_BATCH_TOTAL_TOKENS: int = 2048  # Limits the number of tokens that can be processed in parallel during the generation.



settings = Settings()
