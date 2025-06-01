import json
import os
from pathlib import Path
from pydantic import BaseModel, Field
import librosa

from .config import config


class WordData(BaseModel):
    """Raw word data"""

    name: str
    audios: list[Path]

class WordBank(BaseModel):
    words: list[WordData] = Field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: Path = config.WORD_BANK_FILE):
        """Load word bank from JSON file"""
        if not file_path.exists():
            return cls()

        with open(file_path, "r") as f:
            data = json.load(f)

        words_data = []
        for word_data in data["words"]:
            words_data.append(WordData(
                name=word_data["name"],
                audios=[config.get_absolute_path(Path(audio)) for audio in word_data["audios"]]
            ))

        return cls(words=words_data)

    def save_to_file(self, file_path: Path = config.WORD_BANK_FILE):
        """Save word bank to JSON file"""
        data = {
            "words": [
                {
                    "name": word.name,
                    "audios": [str(config.get_relative_path(audio)) for audio in word.audios]
                }
                for word in self.words
            ]
        }

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

word_bank = WordBank.from_file(config.WORD_BANK_FILE)
