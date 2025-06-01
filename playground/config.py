from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True)

    OPENAI_API_KEY: Optional[str]
    HUGGINGFACE_API_TOKEN: Optional[str]

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

    def get_absolute_path(self, path: Path) -> Path:
        """Convert a path relative to DATA_DIR to an absolute path"""
        if path.is_absolute():
            return path
        return self.DATA_DIR / path

    def get_relative_path(self, path: Path) -> Path:
        """Convert an absolute path to a path relative to DATA_DIR"""
        try:
            return path.relative_to(self.DATA_DIR)
        except ValueError:
            # If the path is not relative to DATA_DIR, return it as is
            return path

config = Config()
