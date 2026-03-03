"""
ASR (Automatic Speech Recognition) module for Notely.
"""

from notely.asr.base import ASRBackend, ASRResult, TranscriptSegment
from notely.asr.funasr import FunASRBackend
from notely.asr.whisper import WhisperBackend

__all__ = [
    "ASRBackend",
    "ASRResult",
    "FunASRBackend",
    "TranscriptSegment",
    "WhisperBackend",
]
