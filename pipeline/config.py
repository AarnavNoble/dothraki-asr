"""Central configuration for the Dothraki ASR pipeline."""

from enum import StrEnum
from pathlib import Path


class Strategy(StrEnum):
    PHONEME = "phoneme"
    EMBEDDING = "embedding"
    DTW = "dtw"
    FINETUNE = "finetune"
    ENSEMBLE = "ensemble"

# Project root
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw"
PROCESSED_AUDIO_DIR = DATA_DIR / "processed"
LEXICON_DIR = DATA_DIR / "lexicon"
RESULTS_DIR = DATA_DIR / "results"
FEATURES_DIR = DATA_DIR / "features"
DIALOGUE_DIR = DATA_DIR / "dialogue"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = ROOT_DIR / "models"
FINETUNE_MODEL_DIR = MODELS_DIR / "whisper-tiny-dothraki"

# Audio settings
SAMPLE_RATE = 16000  # Whisper expects 16kHz
AUDIO_FORMAT = "wav"

# Whisper model sizes available for comparison
WHISPER_MODELS = ["tiny", "base", "small", "medium"]
DEFAULT_WHISPER_MODEL = "small"

# Lexicon
LEXICON_FILE = LEXICON_DIR / "dothraki_lexicon.json"

# Matching engine
TOP_K_MATCHES = 5  # Number of candidate matches to return
