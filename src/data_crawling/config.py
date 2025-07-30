from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = str(Path(__file__).parent.parent.parent)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR, env_file_encoding="utf-8")

    MONGO_DATABASE_HOST: str = (
        "mongodb://mathpal:mathpal@mongo:27017"
    )
    MONGO_DATABASE_NAME: str = "mathpal"



settings = Settings()
