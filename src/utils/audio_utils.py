"""
Audio Utility Functions

Helper functions for audio processing.
"""

from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np
from scipy import signal


def load_audio(
    audio_path: Union[str, Path], 
    sample_rate: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return as numpy array.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate (default: 16000 for Whisper)
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    import soundfile as sf
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Read audio file
    audio, sr = sf.read(str(audio_path))
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if necessary
    if sr != sample_rate:
        # Calculate number of samples after resampling
        num_samples = int(len(audio) * sample_rate / sr)
        audio = signal.resample(audio, num_samples)
    
    return audio.astype(np.float32), sample_rate


def split_into_chunks(
    audio: np.ndarray, 
    sample_rate: int, 
    chunk_duration: float
) -> Iterator[Tuple[np.ndarray, float, float]]:
    """
    Split audio into chunks of specified duration.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate of the audio
        chunk_duration: Duration of each chunk in seconds
        
    Yields:
        Tuple of (audio_chunk, start_time, end_time)
    """
    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = len(audio)
    
    for i in range(0, total_samples, chunk_samples):
        chunk = audio[i:i + chunk_samples]
        start_time = i / sample_rate
        end_time = min((i + chunk_samples) / sample_rate, total_samples / sample_rate)
        
        # Only yield non-empty chunks
        if len(chunk) > 0:
            yield chunk, start_time, end_time


def get_audio_duration(audio_path: Union[str, Path]) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    import soundfile as sf
    
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    info = sf.info(str(audio_path))
    return info.duration


def convert_audio_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: str = "wav"
) -> Path:
    """
    Convert audio file to a different format.
    
    Useful for handling various input formats (mp3, m4a, etc.)
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output file
        output_format: Target format (default: wav)
        
    Returns:
        Path to the converted file
    """
    from pydub import AudioSegment
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load and export
    audio = AudioSegment.from_file(str(input_path))
    audio.export(str(output_path), format=output_format)
    
    return output_path
