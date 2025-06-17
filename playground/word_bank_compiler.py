from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt

from .config import config
from .word_bank import WordData

from playground.word_bank import WordBank

class ComputedWord:
    """Word with computed audio features and analysis"""
    name: str
    audios: list[Path]
    mel_spectrograms: list[np.ndarray]
    mel_centroid: np.ndarray | None
    embeddings: list[np.ndarray]
    embedding_centroid: np.ndarray | None
    duration_frames: int

    def __init__(self, word_data: WordData):
        self.name = word_data.name
        self.audios: list[Path] = word_data.audios.copy()
        self.mel_spectrograms: list[np.ndarray] = []
        self.mel_centroid: np.ndarray | None = None
        self.embeddings: list[np.ndarray] = []  # Audio embeddings
        self.embedding_centroid: np.ndarray | None = None  # Mean embedding
        self.duration_frames: int = 0  # Average duration in mel frames


class WordBankCompiler:
    """Compiles WordData into ComputedWord with audio features"""

    def __init__(
        self,
        sr: int = config.SAMPLE_RATE,
        n_mels: int = config.N_MELS,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def compile_word(self, word_data: WordData) -> ComputedWord:
        """Compile a single word with all its features"""
        computed_word = ComputedWord(word_data)
        self._compute_mel_spectrograms(computed_word)
        self._compute_centroids(computed_word)
        self._compute_embeddings(computed_word)
        self._compute_duration(computed_word)
        return computed_word

    def compile_word_bank(self, word_bank: WordBank) -> list[ComputedWord]:
        """Compile entire word bank"""
        compiled_words = []
        for word_data in word_bank.words:
            compiled_word = self.compile_word(word_data)
            compiled_words.append(compiled_word)

        # Save mel spectrogram images for debugging
        self._save_mel_spectrogram_images(compiled_words)

        return compiled_words

    def _compute_mel_spectrograms(self, computed_word: ComputedWord):
        """Compute mel spectrograms for all audio files"""
        computed_word.mel_spectrograms.clear()

        for audio_path in computed_word.audios:
            try:
                # Load audio file
                y, sr = librosa.load(config.DATA_DIR/audio_path, sr=self.sr)

                # Compute mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length
                )

                # Convert to decibel scale for better representation
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                computed_word.mel_spectrograms.append(mel_spec_db)

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

    def _compute_centroids(self, computed_word: ComputedWord):
        """Compute mel spectrogram centroid (mean across all spectrograms)"""
        if not computed_word.mel_spectrograms:
            computed_word.mel_centroid = None
            return

        # Find the minimum length across all spectrograms for consistent shape
        min_time_frames = min(spec.shape[1] for spec in computed_word.mel_spectrograms)

        # Truncate all spectrograms to the same length
        truncated_specs = [spec[:, :min_time_frames] for spec in computed_word.mel_spectrograms]

        # Stack and compute mean
        stacked_specs = np.stack(truncated_specs, axis=0)
        computed_word.mel_centroid = np.mean(stacked_specs, axis=0)

    def _compute_embeddings(self, computed_word: ComputedWord):
        """Compute audio embeddings (placeholder for your embedding method)"""
        # Implement your embedding computation here
        # This could use pre-trained models, MFCC features, etc.
        pass

    def _compute_duration(self, computed_word: ComputedWord):
        """Compute average duration in frames"""
        if computed_word.mel_spectrograms:
            durations = [spec.shape[1] for spec in computed_word.mel_spectrograms]
            computed_word.duration_frames = int(np.mean(durations))

    def _save_mel_spectrogram_images(self, computed_words: list[ComputedWord]):
        """Save mel spectrogram images for debugging purposes"""
        # Create debug directory if it doesn't exist
        debug_dir = config.DATA_DIR / "debug_mel_spectrograms"
        debug_dir.mkdir(exist_ok=True)

        for computed_word in computed_words:
            # Create word directory
            word_dir = debug_dir / computed_word.name
            word_dir.mkdir(exist_ok=True)

            # Save individual mel spectrograms
            for i, mel_spec in enumerate(computed_word.mel_spectrograms):
                plt.figure(figsize=(10, 4))
                plt.imshow(mel_spec, aspect='auto', origin='lower')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"{computed_word.name} - Sample {i+1}")
                plt.tight_layout()
                plt.savefig(word_dir / f"{computed_word.name}_{i+1}.png")
                plt.close()

            # Save centroid (mean) mel spectrogram if it exists
            if computed_word.mel_centroid is not None:
                plt.figure(figsize=(10, 4))
                plt.imshow(computed_word.mel_centroid, aspect='auto', origin='lower')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f"{computed_word.name} - Mean")
                plt.tight_layout()
                plt.savefig(word_dir / f"{computed_word.name}_mean.png")
                plt.close()
