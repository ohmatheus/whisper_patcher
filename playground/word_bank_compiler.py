
import numpy as np

from .config import config
from .word_bank import WordData


class ComputedWord:
    """Word with computed audio features and analysis"""

    def __init__(self, word_data: WordData):
        self.name = word_data.name
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
        return compiled_words

    def _compute_mel_spectrograms(self, computed_word: ComputedWord):
        """Compute mel spectrograms for all audio files"""
        computed_word.mel_spectrograms.clear()

    def _compute_centroids(self, computed_word: ComputedWord):
        """Compute mel spectrogram centroid"""

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
