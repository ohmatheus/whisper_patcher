import logging

import librosa
import torch
from config import config
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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

            # Convert to float32 and create input features
            inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            attention_mask = torch.ones_like(input_features[:, :, 0]).to(device)

            forced_decoder_ids = processor.get_decoder_prompt_ids(language="fr", task="transcribe")
            predicted_ids = model.generate(
                input_features, attention_mask=attention_mask, forced_decoder_ids=forced_decoder_ids, max_length=448
            )

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            logger.info(f"Transcription: {transcription}")

        except Exception as e:
            logger.error(f"Error processing {mp3_file.name}: {e}")


def main() -> None:
    logger.info("Whisper Sound Bank - Starting application...")
    logger.info(f"Using data directory: {config.DATA_DIR}")
    logger.info(f"Word bank file: {config.WORD_BANK_FILE}")

    test_cuda_gpu()
    test_whisper_transcription()


if __name__ == "__main__":
    main()
