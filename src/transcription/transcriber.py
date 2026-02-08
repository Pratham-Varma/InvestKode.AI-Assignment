"""
Streaming Transcription Module

Implement your audio-to-text transcription pipeline here.

Requirements:
- Process audio in chunks (streaming/near-real-time)
- Use open-source ASR (Whisper) or speech API
- Handle Indian accents and domain-specific terminology

Suggested Libraries:
- openai-whisper: Local Whisper model
- faster-whisper: Optimized Whisper implementation
- speechrecognition: Simple API wrapper
"""

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator


@dataclass
class TranscriptChunk:
    """Represents a chunk of transcribed audio."""
    
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    confidence: float = 1.0
    
    # Optional: speaker info for diarization bonus
    speaker: str | None = None


class StreamingTranscriber:
    """
    Streaming audio transcription.
    
    TODO: Implement this class with your chosen ASR approach.
    
    Example usage:
        transcriber = StreamingTranscriber()
        async for chunk in transcriber.process_audio("audio.wav", chunk_duration=5.0):
            print(f"[{chunk.start_time:.1f}s] {chunk.text}")
    """
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        import logging
        from src.utils.device_utils import get_device, get_compute_type
        
        self.model_name = model_name
        self.device = get_device()
        self.compute_type = get_compute_type(self.device)
        self.model = None
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"Initializing Whisper transcriber "
            f"(model={model_name}, device={self.device}, compute_type={self.compute_type})"
        )
        
        # Lazy loading - model will be loaded on first use
        
    def _load_model(self):
        """Load the Whisper model (lazy loading for better memory management)."""
        if self.model is None:
            from faster_whisper import WhisperModel
            
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,  # Use default cache
            )
            
            self.logger.info("Model loaded successfully")
    
    async def process_audio(
        self, 
        audio_path: Path | str, 
        chunk_duration: float = 5.0
    ) -> AsyncIterator[TranscriptChunk]:
        """
        Process an audio file in chunks and yield transcribed text.
        
        Args:
            audio_path: Path to the audio file
            chunk_duration: Duration of each chunk in seconds
            
        Yields:
            TranscriptChunk objects with transcribed text and metadata
        """
        import asyncio
        from src.utils.audio_utils import load_audio, split_into_chunks
        
        # Ensure model is loaded
        self._load_model()
        
        self.logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio
        audio, sample_rate = load_audio(audio_path, sample_rate=16000)
        
        # Split into chunks
        chunks = list(split_into_chunks(audio, sample_rate, chunk_duration))
        self.logger.info(f"Split audio into {len(chunks)} chunks")
        
        # Process each chunk
        for i, (audio_chunk, start_time, end_time) in enumerate(chunks):
            # Run transcription in executor to avoid blocking
            result = await asyncio.to_thread(
                self._transcribe_chunk,
                audio_chunk,
                start_time,
                end_time
            )
            
            if result:
                yield result
    
    def _transcribe_chunk(
        self,
        audio_chunk: "np.ndarray",
        start_time: float,
        end_time: float
    ) -> TranscriptChunk | None:
        """
        Transcribe a single audio chunk (synchronous).
        
        Args:
            audio_chunk: Audio data as numpy array
            start_time: Start time of chunk in seconds
            end_time: End time of chunk in seconds
            
        Returns:
            TranscriptChunk or None if transcription fails
        """
        try:
            # Transcribe using faster-whisper
            segments, info = self.model.transcribe(
                audio_chunk,
                language="en",  # Auto-detect or specify
                task="transcribe",
                vad_filter=True,  # Voice activity detection
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                ),
            )
            
            # Combine all segments from this chunk
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            full_text = " ".join(text_parts).strip()
            
            if not full_text:
                self.logger.debug(f"No speech detected in chunk [{start_time:.1f}s - {end_time:.1f}s]")
                return None
            
            return TranscriptChunk(
                text=full_text,
                start_time=start_time,
                end_time=end_time,
                confidence=1.0,  # faster-whisper doesn't provide per-word confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error transcribing chunk: {e}")
            return None
    
    async def process_audio_stream(
        self, 
        audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptChunk]:
        """
        Process a live audio stream (for real-time applications).
        
        Args:
            audio_stream: Async iterator of audio bytes
            
        Yields:
            TranscriptChunk objects
        """
        # TODO: Implement for live audio streaming (bonus)
        raise NotImplementedError("TODO: Implement live audio streaming")
