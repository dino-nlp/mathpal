from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    # Hugging Face config
    HUGGINGFACE_BASE_MODEL_ID: str = "unsloth/gemma-3n-E4B"
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # Comet config
    COMET_API_KEY: str | None = None
    COMET_WORKSPACE: str | None = None
    COMET_PROJECT: str = "mathpal-gemma3n"

    DATASET_ID: str = "exam-sixth_grade-instruct-dataset"  # Comet artifact containing your fine-tuning dataset (available after generating the instruct dataset).

    # AWS config
    AWS_REGION: str = "eu-central-1"
    AWS_ACCESS_KEY: str | None = None
    AWS_SECRET_KEY: str | None = None
    AWS_ARN_ROLE: str | None = None


settings = Settings()