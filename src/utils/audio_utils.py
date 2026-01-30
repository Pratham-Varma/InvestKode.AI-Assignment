"""
Audio Utility Functions

Helper functions for audio processing.
"""

from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np


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
    # TODO: Implement audio loading
    # 
    # Example with soundfile:
    # 
    # import soundfile as sf
    # audio, sr = sf.read(audio_path)
    # 
    # # Resample if necessary
    # if sr != sample_rate:
    #     # Use librosa or scipy for resampling
    #     pass
    # 
    # # Convert to mono if stereo
    # if len(audio.shape) > 1:
    #     audio = audio.mean(axis=1)
    # 
    # return audio.astype(np.float32), sample_rate
    
    raise NotImplementedError("TODO: Implement audio loading")


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
    # TODO: Implement audio chunking
    # 
    # Example:
    # 
    # chunk_samples = int(chunk_duration * sample_rate)
    # total_samples = len(audio)
    # 
    # for i in range(0, total_samples, chunk_samples):
    #     chunk = audio[i:i + chunk_samples]
    #     start_time = i / sample_rate
    #     end_time = min((i + chunk_samples) / sample_rate, total_samples / sample_rate)
    #     yield chunk, start_time, end_time
    
    raise NotImplementedError("TODO: Implement audio chunking")


def get_audio_duration(audio_path: Union[str, Path]) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    # TODO: Implement
    raise NotImplementedError("TODO: Implement")


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
    # TODO: Implement using pydub
    # 
    # from pydub import AudioSegment
    # audio = AudioSegment.from_file(input_path)
    # audio.export(output_path, format=output_format)
    # return Path(output_path)
    
    raise NotImplementedError("TODO: Implement audio conversion")
