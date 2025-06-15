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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cuda_gpu() -> None:
    logger.info("\n=== CUDA GPU Test for PyTorch ===")

    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {device_count}")

        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            logger.info(
                f"Device {i}: {device_name} (Compute Capability: {device_capability[0]}.{device_capability[1]})"
            )

        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device}")

        logger.info("\nRunning a simple tensor operation on GPU...")
        try:
            x = torch.rand(5, 3).cuda()
            y = torch.rand(5, 3).cuda()

            z = x + y

            logger.info("Tensor operation successful!")
            logger.info(f"Input tensor 1 shape: {x.shape}, device: {x.device}")
            logger.info(f"Input tensor 2 shape: {y.shape}, device: {y.device}")
            logger.info(f"Output tensor shape: {z.shape}, device: {z.device}")

            z_cpu = z.cpu()
            logger.info(f"Sample output values: {z_cpu[0]}")

            logger.info("\nCUDA GPU test completed successfully!")
        except Exception as e:
            logger.info(f"Error during tensor operation: {e}")
    else:
        logger.info("CUDA is not available. PyTorch will run on CPU only.")
        logger.info("If you expected CUDA to be available, please check:")
        logger.info("1. You have an NVIDIA GPU")
        logger.info("2. NVIDIA drivers are installed")
        logger.info("3. CUDA toolkit is installed")
        logger.info("4. PyTorch was installed with CUDA support")

@dataclass
class WordInfo:
    word: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None

def _extract_word_timestamps(
    self,
    token_ids: torch.Tensor,
    token_timestamps: Optional[torch.Tensor] = None,
    scores: Optional[Tuple[torch.Tensor]] = None,
) -> list[WordInfo]:
    # Decode tokens individually
    words_info = []
    current_word = ""
    word_start_time = None
    word_tokens = []

    for i, token_id in enumerate(token_ids):
        token_id_int = int(token_id)

        # Skip special tokens
        if token_id_int in self.processor.tokenizer.all_special_ids:
            continue

        # Decode single token
        token_text = self.processor.decode([token_id_int])

        # Calculate timestamp if available
        if token_timestamps is not None:
            # Whisper's time tokens represent 0.02 second intervals
            timestamp = float(token_timestamps[i]) * 0.02
        else:
            # Fallback: estimate based on position
            timestamp = i * 0.02

        # Calculate confidence if scores are available
        confidence = None
        if scores is not None and i < len(scores):
            # Get the probability of the selected token
            probs = torch.softmax(scores[i][0], dim=-1)
            confidence = float(probs[token_id_int])

        # Check if this token starts a new word (usually starts with space or is the first token)
        if token_text.startswith(" ") or not current_word:
            # Save previous word if exists
            if current_word:
                words_info.append(
                    WordInfo(
                        word=current_word.strip(),
                        start_time=word_start_time,
                        end_time=timestamp,
                        confidence=np.mean(word_tokens) if word_tokens else None,
                    )
                )

            # Start new word
            current_word = token_text
            word_start_time = timestamp
            word_tokens = [confidence] if confidence is not None else []
        else:
            # Continue current word
            current_word += token_text
            if confidence is not None:
                word_tokens.append(confidence)

    # Don't forget the last word
    if current_word:
        words_info.append(
            WordInfo(
                word=current_word.strip(),
                start_time=word_start_time,
                end_time=timestamp,
                confidence=np.mean(word_tokens) if word_tokens else None,
            )
        )

    return words_info

def test_whisper_transcription() -> None:
    logger.info("\n=== Whisper Transcription Test ===")
    logger.info(f"Using Whisper model: {config.WHISPER_MODEL}")

    logger.info("Loading Whisper model...")
    model_name = f"openai/whisper-{config.WHISPER_MODEL}"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded successfully on {device}")

    # Get all mp3 files from the sentences_tests directory
    sentences_dir = config.DATA_DIR / "sentences_tests"
    mp3_files = list(sentences_dir.glob("*.mp3"))
    logger.info(f"Found {len(mp3_files)} mp3 files in {sentences_dir}")

    # Process each mp3 file
    for mp3_file in mp3_files:
        logger.info(f"\nProcessing file: {mp3_file.name}")

        try:
            audio, sampling_rate = librosa.load(mp3_file, sr=config.SAMPLE_RATE)

            inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            attention_mask = torch.ones_like(input_features[:, :, 0]).to(device)

            forced_decoder_ids = processor.get_decoder_prompt_ids(language="fr", task="transcribe", no_timestamps=False)

            # prompt = "Ces mots peuvent apparaitre : distriviche, pokamoungatoa, zuguludivska"
            # prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt").to(device)


            # predicted_ids = model.generate(
            #     input_features,
            #     attention_mask=attention_mask,
            #     forced_decoder_ids=forced_decoder_ids,
            #     max_length=448,
            #     # prompt_ids=prompt_ids,
            #     return_timestamps=True,
            # )

            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features,
                    #attention_mask=attention_mask,
                    return_timestamps=True,
                    return_token_timestamps=True,
                    return_dict_in_generate=True,
                )

            #transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]

            decode = processor.batch_decode(
                predicted_ids.sequences,
                skip_special_tokens=True,
                decode_with_timestamps=True
            )
            transcription_with_timestamps = decode[0]

            logger.info(f"Transcription: {transcription_with_timestamps}")

        except Exception as e:
            logger.error(f"Error processing {mp3_file.name}: {e}")



def test_whisper_with_timestamp_transcription() -> None:
    logger.info("\n=== Whisper Transcription Test ===")
    logger.info(f"Using Whisper model: {config.WHISPER_MODEL}")

    logger.info("Loading Whisper model...")
    model_name = f"openai/whisper-{config.WHISPER_MODEL}"

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded successfully on {device}")

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


    # Process each mp3 file
    for mp3_file in mp3_files:
        logger.info(f"\nProcessing file: {mp3_file.name}")


#        audio = wt.load_audio(str(mp3_file.absolute()), sr=config.SAMPLE_RATE)
        try:
            audio, sr = librosa.load(mp3_file, sr=config.SAMPLE_RATE)

            model = wt.load_model(model_name, device=device)

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

    #test_cuda_gpu()
    #test_whisper_transcription()

    test_whisper_with_timestamp_transcription()


if __name__ == "__main__":
    main()
