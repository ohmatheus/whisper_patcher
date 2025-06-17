import logging
import json
import librosa
import torch
from config import config
from dataclasses import dataclass
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import numpy as np
from typing import Optional, Tuple
import whisper_timestamped as wt

from playground.word_bank import WordBank
from playground.word_bank_compiler import WordBankCompiler
from playground.word_bank_compiler import ComputedWord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WordInfo:
    word: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


def test_whisper_with_timestamp_transcription() -> None:
    logger.info("\n=== Whisper Transcription Test ===")
    logger.info(f"Using Whisper model: {config.WHISPER_MODEL}")

    logger.info("Loading Whisper model...")
    model_name = f"openai/whisper-{config.WHISPER_MODEL}"

    #processor = WhisperProcessor.from_pretrained(model_name)
    #model = WhisperForConditionalGeneration.from_pretrained(model_name)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Model loaded successfully on {device}")
    model = wt.load_model(model_name, device=device)

    # Get all mp3 files from the sentences_tests directory
    sentences_dir = config.DATA_DIR / "sentences_tests"
    mp3_files = list(sentences_dir.glob("*.mp3"))
    logger.info(f"Found {len(mp3_files)} mp3 files in {sentences_dir}")

    # Create output directory if it doesn't exist
    output_dir = config.DATA_DIR / "test_timestamped_output"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    for mp3_file in mp3_files:
        logger.info(f"\nProcessing file: {mp3_file.name}")
        logger.info(f"Full path: {mp3_file}")
        logger.info(f"Absolute path: {mp3_file.absolute()}")
        logger.info(f"File exists: {mp3_file.exists()}")

    word_bank = WordBank.from_file(config.WORD_BANK_FILE)
    compiler = WordBankCompiler()
    computed_words: list[ComputedWord] = compiler.compile_word_bank(word_bank=word_bank)

    # Process each mp3 file
    for mp3_file in mp3_files:
        logger.info(f"\nProcessing file: {mp3_file.name}")

        try:
            audio, sr = librosa.load(mp3_file, sr=config.SAMPLE_RATE)

            result = wt.transcribe(model, audio, language="fr")

            # Log the transcription
            logger.info(f"Transcription: {json.dumps(result, indent=2, ensure_ascii=False)}")

            # Save the result to a JSON file with the original filename
            output_file = output_dir / f"{mp3_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved transcription to {output_file}")

        except Exception as e:
            logger.error(f"Error processing {mp3_file.name}: {e}")


def main() -> None:
    logger.info("Whisper Sound Bank - Starting application...")
    logger.info(f"Using data directory: {config.DATA_DIR}")
    logger.info(f"Word bank file: {config.WORD_BANK_FILE}")

    test_whisper_with_timestamp_transcription()


if __name__ == "__main__":
    main()
