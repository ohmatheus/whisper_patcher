from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True)

    OPENAI_API_KEY: str | None
    HUGGINGFACE_API_TOKEN: str | None

    DATA_DIR: Path
    WORD_BANK_DIR: Path
    WORD_BANK_FILE: Path
    VECTOR_DB_PATH: Path

    WHISPER_MODEL: str
    EMBEDDING_MODEL: str

    SAMPLE_RATE: int
    N_MELS: int
    N_FFT: int
    HOP_LENGTH: int


config = Config()
