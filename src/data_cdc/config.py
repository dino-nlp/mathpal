from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    MONGO_DATABASE_HOST: str = "mongodb://mathpal:mathpal@127.0.0.1:27017"
    MONGO_DATABASE_NAME: str = "mathpal"

    RABBITMQ_HOST: str = "localhost"  # or localhost if running outside Docker
    RABBITMQ_PORT: int = 5672
    RABBITMQ_DEFAULT_USERNAME: str = "guest"
    RABBITMQ_DEFAULT_PASSWORD: str = "guest"
    RABBITMQ_QUEUE_NAME: str = "default"


settings = Settings()