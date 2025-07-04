import json
from pathlib import Path

from playground.config import config
from pydantic import BaseModel, Field


class WordData(BaseModel):
    name: str
    audios: list[Path]


class WordBank(BaseModel):
    words: list[WordData] = Field(default_factory=list)

    @classmethod
    def from_file(cls, file_path: Path = config.WORD_BANK_FILE) -> "WordBank":
        if not file_path.exists():
            return cls()

        with open(file_path) as f:
            data = json.load(f)

        words_data = []
        for word_data in data["words"]:
            words_data.append(WordData(name=word_data["name"], audios=[Path(audio) for audio in word_data["audios"]]))

        return cls(words=words_data)

    def save_to_file(self, file_path: Path = config.WORD_BANK_FILE) -> None:
        serializable_words = []
        for word in self.words:
            serializable_words.append(WordData(name=word.name, audios=[audio for audio in word.audios]))

        serializable_bank = WordBank(words=serializable_words)

        data = serializable_bank.model_dump()

        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
