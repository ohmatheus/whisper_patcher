import json
import os
from pathlib import Path
from pydantic import BaseModel, Field
import librosa

from config import config

class WordData(BaseModel):
    name: str
    audios: list[Path]

class WordBank(BaseModel):
    words: list[WordData] = Field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: Path = config.WORD_BANK_FILE):
        if not file_path.exists():
            return cls()

        with open(file_path, "r") as f:
            data = json.load(f)

        words_data = []
        for word_data in data["words"]:
            words_data.append(WordData(
                name=word_data["name"],
                audios=[Path(audio) for audio in word_data["audios"]]
            ))

        return cls(words=words_data)

    def save_to_file(self, file_path: Path = config.WORD_BANK_FILE):
        serializable_words = []
        for word in self.words:
            serializable_words.append(WordData(
                name=word.name,
                audios=[audio for audio in word.audios]
            ))

        serializable_bank = WordBank(words=serializable_words)

        data = serializable_bank.model_dump()

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

word_bank = WordBank.from_file(config.WORD_BANK_FILE)
