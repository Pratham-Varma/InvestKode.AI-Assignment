"""
Tests for audio utilities.
"""

import pytest
import numpy as np
from pathlib import Path
from src.utils.audio_utils import split_into_chunks


class TestAudioChunking:
    """Tests for audio chunking functionality."""
    
    def test_split_into_chunks(self):
        """Test splitting audio into chunks."""
        # Create fake audio: 10 seconds at 16kHz
        sample_rate = 16000
        duration = 10.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Split into 2-second chunks
        chunk_duration = 2.0
        chunks = list(split_into_chunks(audio, sample_rate, chunk_duration))
        
        # Should have 5 chunks (10s / 2s = 5)
        assert len(chunks) == 5
        
        # Verify first chunk
        chunk, start, end = chunks[0]
        assert start == 0.0
        assert end == 2.0
        assert len(chunk) == int(sample_rate * chunk_duration)
        
        # Verify last chunk
        chunk, start, end = chunks[-1]
        assert start == 8.0
        assert end == 10.0
    
    def test_split_uneven_chunks(self):
        """Test splitting when duration doesn't divide evenly."""
        sample_rate = 16000
        duration = 7.5  # 7.5 seconds
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        chunk_duration = 3.0
        chunks = list(split_into_chunks(audio, sample_rate, chunk_duration))
        
        # Should have 3 chunks: 0-3, 3-6, 6-7.5
        assert len(chunks) == 3
        
        # Last chunk should be shorter
        chunk, start, end = chunks[-1]
        assert end == 7.5
        assert len(chunk) < int(sample_rate * chunk_duration)
