"""
FunASR backend for ASR - Alibaba's high-quality Chinese ASR model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from notely.asr.base import ASRBackend, ASRResult, TranscriptSegment


class FunASRBackend(ASRBackend):
    """
    FunASR backend using Paraformer model.

    FunASR provides state-of-the-art ASR for Chinese with:
    - High accuracy (CER < 3%)
    - Timestamp support
    - Speaker diarization
    - Punctuation restoration

    Args:
        model: Model name or path. Defaults to Paraformer-large.
        device: Device to use ("cuda" or "cpu").
        use_vad: Whether to use voice activity detection.
        use_punc: Whether to use punctuation restoration.
        use_spk: Whether to use speaker diarization.
    """

    def __init__(
        self,
        model: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        device: str = "cuda",
        use_vad: bool = True,
        use_punc: bool = True,
        use_spk: bool = False,
    ):
        self.model_name = model
        self.device = device
        self.use_vad = use_vad
        self.use_punc = use_punc
        self.use_spk = use_spk
        self._model = None

    def _load_model(self) -> Any:
        """Lazy load the FunASR model."""
        try:
            from funasr import AutoModel  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "FunASR is not installed. Install it with: pip install funasr modelscope"
            )

        # Use single model with punc_model and vad_model parameters
        punc_model = (
            "iic/punc_ct-transformer_cn-en-common-vocab471067-large" if self.use_punc else None
        )
        vad_model = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch" if self.use_vad else None

        return AutoModel(
            model=self.model_name,
            punc_model=punc_model,
            vad_model=vad_model,
            device=self.device,
        )

    @property
    def model(self) -> Any:
        """Get the model (lazy loading)."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def transcribe(self, audio_path: Path | str) -> ASRResult:
        """
        Transcribe audio using FunASR.

        Args:
            audio_path: Path to audio file.

        Returns:
            ASRResult with transcribed segments.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result = self.model.generate(input=str(audio_path))

        segments = []

        # FunASR with VAD returns results with sentence-level timestamps
        # Each item in result contains a sentence with its timestamp
        for item in result:
            # Handle different result formats
            if isinstance(item, dict):
                text = item.get("text", "")
                timestamp = item.get("timestamp", [])

                # FunASR returns timestamps as list of [start, end] pairs for each word
                # We need to extract sentence-level timing
                if timestamp and isinstance(timestamp, list) and len(timestamp) > 0:
                    # Get the first and last timestamp for sentence boundaries
                    if isinstance(timestamp[0], (list, tuple)) and len(timestamp[0]) >= 2:
                        start_time = timestamp[0][0] / 1000.0  # Convert ms to seconds
                        end_time = (
                            timestamp[-1][1] / 1000.0
                            if len(timestamp) > 1
                            else timestamp[0][1] / 1000.0
                        )
                    else:
                        start_time = 0.0
                        end_time = 0.0
                else:
                    start_time = 0.0
                    end_time = 0.0

                # Split long text into smaller segments if no detailed timestamps
                if text and len(text) > 100 and start_time == end_time:
                    # Split by punctuation for better granularity
                    import re

                    sentences = re.split(r"([。！？；])", text)
                    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2] + [""])]
                    sentences = [s.strip() for s in sentences if s.strip()]

                    for i, sent in enumerate(sentences):
                        segments.append(
                            TranscriptSegment(
                                text=sent,
                                start_time=start_time + i * 5.0,  # Estimate
                                end_time=start_time + (i + 1) * 5.0,
                                confidence=item.get("confidence", 1.0),
                                speaker_id=item.get("spk", None),
                            )
                        )
                elif text:
                    segments.append(
                        TranscriptSegment(
                            text=text,
                            start_time=start_time,
                            end_time=end_time,
                            confidence=item.get("confidence", 1.0),
                            speaker_id=item.get("spk", None),
                        )
                    )

        # Calculate total duration
        duration = max((seg.end_time for seg in segments), default=0.0)

        return ASRResult(
            segments=segments,
            language="zh",
            duration=duration,
            metadata={
                "backend": "funasr",
                "model": self.model_name,
                "segment_count": len(segments),
            },
        )

    def is_available(self) -> bool:
        """Check if FunASR is available."""
        try:
            import funasr  # noqa: F401

            return True
        except ImportError:
            return False


class FunASRStreaming:
    """
    Streaming ASR for real-time transcription.

    Useful for live lectures or long recordings that need
    incremental processing.
    """

    def __init__(
        self, model: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    ):
        self.model_name = model
        self._model = None

    def _load_model(self) -> Any:
        """Load streaming model."""
        try:
            from funasr import AutoModel
        except ImportError:
            raise ImportError("FunASR is not installed")

        return AutoModel(model=self.model_name, model_revision="v2.0.4")

    @staticmethod
    def transcribe_chunk(audio_chunk: bytes) -> str:
        """Transcribe a chunk of audio data."""
        # Implementation for streaming
        raise NotImplementedError("Streaming transcription not yet implemented")
