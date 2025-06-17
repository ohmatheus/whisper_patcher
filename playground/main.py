import logging
import json
import librosa
import torch
import time
import matplotlib.pyplot as plt
from playground.config import config
from dataclasses import dataclass
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import whisper_timestamped as wt
from scipy.spatial.distance import euclidean, cityblock, cosine
from dtaidistance import dtw

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


def extract_mel_spectrogram_for_word(audio: np.ndarray, sr: int, start_time: float, end_time: float, word_text: str = "unknown") -> np.ndarray:
    """Extract mel spectrogram for a specific word based on its timestamps"""
    # Convert time to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract the audio segment for the word
    word_audio = audio[start_sample:end_sample]

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=word_audio,
        sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )

    # Convert to decibel scale
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Frequency wise normalization
    #mel_spec = (mel_spec - mel_spec.mean(axis=1, keepdims=True)) / mel_spec.std(axis=1, keepdims=True)

    # Save mel spectrogram image for debugging
    debug_dir = config.DATA_DIR / "debug_mel_spectrograms"
    debug_dir.mkdir(exist_ok=True)

    # Create word directory
    word_dir = debug_dir / word_text
    word_dir.mkdir(exist_ok=True)

    # Generate a unique filename based on timestamp
    timestamp = int(time.time() * 1000)

    # Save the mel spectrogram image
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{word_text} - {timestamp}")
    plt.tight_layout()
    plt.savefig(word_dir / f"{word_text}_{timestamp}.png")
    plt.close()

    return mel_spec


def find_least_confident_words(transcription: Dict[str, Any], threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Find words with confidence below the threshold and merge consecutive ones"""
    result = []

    # Iterate through each segment
    for segment in transcription["segments"]:
        segment_id = segment["id"]
        low_confidence_words = []

        # Find all low confidence words in this segment
        for word in segment["words"]:
            if word["confidence"] < threshold:
                # Create a copy of the word dict and add segment_id
                word_info = word.copy()
                word_info["segment_id"] = segment_id
                low_confidence_words.append(word_info)

        # Process consecutive low confidence words
        i = 0
        while i < len(low_confidence_words):
            # Start with the current word
            current_word = low_confidence_words[i]

            # Check if there are consecutive low confidence words
            if i + 1 < len(low_confidence_words):
                next_word = low_confidence_words[i + 1]

                # Check if words are consecutive (end time of current = start time of next)
                if abs(current_word["end"] - next_word["start"]) < 0.1:  # Small threshold for floating point comparison
                    # Merge words
                    merged_word = {
                        "text": current_word["text"] + " " + next_word["text"],
                        "start": current_word["start"],
                        "end": next_word["end"],
                        "confidence": (current_word["confidence"] + next_word["confidence"]) / 2,  # Average confidence
                        "segment_id": segment_id,
                        "is_merged": True,
                        "merged_words": [current_word, next_word]
                    }

                    # Skip the next word since we've merged it
                    i += 2

                    # Check if there are more consecutive words to merge
                    j = i
                    while j < len(low_confidence_words):
                        next_word = low_confidence_words[j]
                        last_word = merged_word["merged_words"][-1]

                        if abs(last_word["end"] - next_word["start"]) < 0.1:
                            # Add this word to the merged group
                            merged_word["merged_words"].append(next_word)
                            merged_word["text"] += " " + next_word["text"]
                            merged_word["end"] = next_word["end"]
                            # Update average confidence
                            total_confidence = sum(w["confidence"] for w in merged_word["merged_words"])
                            merged_word["confidence"] = total_confidence / len(merged_word["merged_words"])
                            j += 1
                            i = j  # Update outer loop counter
                        else:
                            break

                    result.append(merged_word)
                    continue

            # If no merging occurred, add the single word
            result.append(current_word)
            i += 1

    return result



def compare_with_computed_words(word_mel_spec: np.ndarray, computed_words: List[ComputedWord]) -> Tuple[Optional[ComputedWord], float]:
    """Compare word mel spectrogram with computed words using DTW

    Args:
        word_mel_spec: Mel spectrogram of the word to compare
        computed_words: List of computed words to compare against
    """
    best_match = None
    best_distance = float('inf')

    for computed_word in computed_words:
        # Skip words with no mel spectrograms
        if not computed_word.mel_spectrograms:
            continue

        # First check if we can compare with individual mel spectrograms
        word_best_distance = float('inf')

        # Compare with each individual mel spectrogram
        for sub_mel_spec in computed_word.mel_spectrograms:
            # Ensure the mel spectrograms have the same number of mel bins
            if word_mel_spec.shape[0] != sub_mel_spec.shape[0]:
                continue

            # For DTW with dtaidistance, we need to flatten the 2D arrays to 1D
            # or compute DTW for each frequency bin and average the distances
            total_distance = 0.0
            valid_comparisons = 0
            
            # Compare each frequency bin (row) separately
            for freq_bin in range(word_mel_spec.shape[0]):
                word_series = word_mel_spec[freq_bin, :]
                sub_series = sub_mel_spec[freq_bin, :]
                
                # Skip if either series is empty or has length < 2
                if len(word_series) < 2 or len(sub_series) < 2:
                    continue
                    
                try:
                    bin_distance = dtw.distance(word_series, sub_series)
                    total_distance += bin_distance
                    valid_comparisons += 1
                except Exception as e:
                    # Skip this frequency bin if DTW fails
                    continue
            
            # Calculate average distance across frequency bins
            if valid_comparisons > 0:
                distance = total_distance / valid_comparisons
                
                if distance < word_best_distance:
                    word_best_distance = distance

        # If we found a good match in the sub spectrograms, use that
        if word_best_distance < float('inf'):
            if word_best_distance < best_distance:
                best_distance = word_best_distance
                best_match = computed_word
        # Fallback to centroid comparison if needed
        elif computed_word.mel_centroid is not None:
            # Ensure the mel spectrogram have the same number of mel bins
            if word_mel_spec.shape[0] != computed_word.mel_centroid.shape[0]:
                continue

            # Compare each frequency bin separately
            total_distance = 0.0
            valid_comparisons = 0
            
            for freq_bin in range(word_mel_spec.shape[0]):
                word_series = word_mel_spec[freq_bin, :]
                centroid_series = computed_word.mel_centroid[freq_bin, :]
                
                # Skip if either series is empty or has length < 2
                if len(word_series) < 2 or len(centroid_series) < 2:
                    continue
                    
                try:
                    bin_distance = dtw.distance(word_series, centroid_series)
                    total_distance += bin_distance
                    valid_comparisons += 1
                except Exception as e:
                    # Skip this frequency bin if DTW fails
                    continue
            
            # Calculate average distance across frequency bins            
            if valid_comparisons > 0:
                distance = total_distance / valid_comparisons
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = computed_word

    return best_match, best_distance


def find_best_sub_match(word_mel_spec: np.ndarray, computed_word: ComputedWord) -> Tuple[int, float]:
    """Find the best match among sub mel spectrograms

    Args:
        word_mel_spec: Mel spectrogram of the word to compare
        computed_word: Computed word to compare against
    """
    best_sub_idx = -1
    best_sub_distance = float('inf')

    for i, sub_mel_spec in enumerate(computed_word.mel_spectrograms):
        # Ensure the mel spectrograms have the same number of mel bins
        if word_mel_spec.shape[0] != sub_mel_spec.shape[0]:
            continue

        # Compare each frequency bin separately
        total_distance = 0.0
        valid_comparisons = 0
        
        for freq_bin in range(word_mel_spec.shape[0]):
            word_series = word_mel_spec[freq_bin, :]
            sub_series = sub_mel_spec[freq_bin, :]
            
            # Skip if either series is empty or has length < 2
            if len(word_series) < 2 or len(sub_series) < 2:
                continue
                
            try:
                bin_distance = dtw.distance(word_series, sub_series)
                total_distance += bin_distance
                valid_comparisons += 1
            except Exception as e:
                # Skip this frequency bin if DTW fails
                continue
        
        # Calculate average distance across frequency bins
        if valid_comparisons > 0:
            distance = total_distance / valid_comparisons
            
            if distance < best_sub_distance:
                best_sub_distance = distance
                best_sub_idx = i

    return best_sub_idx, best_sub_distance


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
            #logger.info(f"Transcription: {json.dumps(result, indent=2, ensure_ascii=False)}")

            # Save the result to a JSON file with the original filename
            output_file = output_dir / f"{mp3_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved transcription to {output_file}")

            # Find least confident words in the transcription
            least_confident_words = find_least_confident_words(result, threshold=0.7)

            # Count individual and merged words
            individual_words = [w for w in least_confident_words if not w.get("is_merged", False)]
            merged_words = [w for w in least_confident_words if w.get("is_merged", False)]
            logger.info(f"Found {len(least_confident_words)} least confident words: {len(individual_words)} individual and {len(merged_words)} merged")

            # Process each least confident word
            for word_info in least_confident_words:
                word_text = word_info["text"]
                start_time = word_info["start"]
                end_time = word_info["end"]
                confidence = word_info["confidence"]
                segment_id = word_info["segment_id"]

                word_type = "merged" if word_info.get("is_merged", False) else "individual"
                logger.info(f"Processing {word_type} least confident word: '{word_text}' (confidence: {confidence:.3f})")

                # Extract mel spectrogram for the word
                word_mel_spec = extract_mel_spectrogram_for_word(audio, sr, start_time, end_time, word_text)

                # Compare with computed words using DTW
                best_match, best_distance = compare_with_computed_words(word_mel_spec, computed_words)

                # If a match is found
                if best_match and best_distance < 200:  # Threshold can be adjusted based on testing
                    logger.info(f"Found potential match: '{best_match.name}' with distance: {best_distance:.2f}")

                    # Find best sub match
                    #best_sub_idx, best_sub_distance = find_best_sub_match(word_mel_spec, best_match)

                    if True:# best_sub_idx >= 0 and best_sub_distance < best_distance:
                        #logger.info(f"YAHOUUU! Found better match in sub mel spectrogram {best_sub_idx+1} with distance: {best_sub_distance:.2f}")

                        # Replace the word in the transcription
                        if word_info.get("is_merged", False):
                            # For merged words, we need to find and replace each original word
                            for original_word in word_info["merged_words"]:
                                # Find the index of the original word in the segment
                                for i, segment_word in enumerate(result["segments"][segment_id]["words"]):
                                    if (segment_word["start"] == original_word["start"] and 
                                        segment_word["end"] == original_word["end"]):
                                        # Replace the word
                                        result["segments"][segment_id]["words"][i]["text"] = best_match.name
                                        break
                        else:
                            # For single words, use the original logic
                            for i, segment_word in enumerate(result["segments"][segment_id]["words"]):
                                if (segment_word["start"] == word_info["start"] and 
                                    segment_word["end"] == word_info["end"]):
                                    # Replace the word
                                    result["segments"][segment_id]["words"][i]["text"] = best_match.name
                                    break

                        # Update the full text
                        full_text = ""
                        for segment in result["segments"]:
                            segment_text = " ".join(word["text"] for word in segment["words"])
                            full_text += " " + segment_text if full_text else segment_text
                        result["text"] = full_text

                        logger.info(f"Replaced '{word_text}' with '{best_match.name}'")
                        logger.info(f"Updated transcription: {result['text']}")

                        # Save the updated result
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        logger.info(f"Saved updated transcription to {output_file}")

        except Exception as e:
            logger.error(f"Error processing {mp3_file.name}: {e}")


def main() -> None:
    logger.info("Whisper Sound Bank - Starting application...")
    logger.info(f"Using data directory: {config.DATA_DIR}")
    logger.info(f"Word bank file: {config.WORD_BANK_FILE}")
    logger.info("Using dtaidistance for DTW calculations")

    test_whisper_with_timestamp_transcription()


if __name__ == "__main__":
    main()