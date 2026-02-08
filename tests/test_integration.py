"""
Integration tests for the complete pipeline.
Run with: pytest tests/test_integration.py -v
"""

import pytest
import os
from pathlib import Path


class TestEnvironmentSetup:
    """Test environment configuration."""
    
    def test_imports(self):
        """Test that all core modules can be imported."""
        from src.transcription.transcriber import StreamingTranscriber, TranscriptChunk
        from src.insights.detector import InsightDetector, Insight, InsightType
        from src.streaming.streamer import ConsoleStreamer
        from src.utils.audio_utils import load_audio, split_into_chunks
        from src.utils.device_utils import get_device
        
        assert StreamingTranscriber is not None
        assert InsightDetector is not None
        assert ConsoleStreamer is not None
    
    def test_device_detection(self):
        """Test GPU/CPU detection."""
        from src.utils.device_utils import get_device
        
        device = get_device()
        assert device in ["cuda", "cpu"]


class TestTranscriptionPipeline:
    """Test transcription functionality."""
    
    def test_transcriber_initialization(self):
        """Test transcriber can be initialized."""
        from src.transcription.transcriber import StreamingTranscriber
        
        transcriber = StreamingTranscriber(model_name="tiny")
        assert transcriber.model_name == "tiny"
        assert transcriber.device in ["cuda", "cpu"]


class TestInsightDetection:
    """Test insight detection functionality."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        from src.insights.detector import InsightDetector
        
        detector = InsightDetector(use_llm=False)
        assert detector is not None
        
    @pytest.mark.asyncio
    async def test_rule_based_detection(self):
        """Test rule-based insight detection."""
        from src.insights.detector import InsightDetector, InsightType
        from src.transcription.transcriber import TranscriptChunk
        
        detector = InsightDetector(use_llm=False)
        
        # Test revenue detection
        chunk = TranscriptChunk(
            text="Our revenue grew by 15 percent to 500 crores this quarter.",
            start_time=0.0,
            end_time=5.0
        )
        
        result = await detector.analyze(chunk)
        
        # Should detect revenue and growth
        assert len(result.insights) > 0
        has_revenue = any(i.type == InsightType.REVENUE for i in result.insights)
        has_growth = any(i.type == InsightType.GROWTH for i in result.insights)
        
        assert has_revenue or has_growth


class TestAPIEndpoints:
    """Test FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_status_endpoint_not_found(self, client):
        """Test status endpoint with non-existent job."""
        response = client.get("/status/nonexistent")
        assert response.status_code == 404


# Add this if you have sample audio files
@pytest.mark.integration
@pytest.mark.skipif(
    not Path("data/samples").exists() or not list(Path("data/samples").glob("*.wav")),
    reason="No sample audio files found"
)
class TestEndToEnd:
    """End-to-end integration tests (requires sample audio)."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete pipeline with sample audio."""
        from src.transcription.transcriber import StreamingTranscriber
        from src.insights.detector import InsightDetector
        
        # Find first sample audio file
        sample_files = list(Path("data/samples").glob("*.wav"))
        if not sample_files:
            pytest.skip("No sample audio files")
        
        audio_file = sample_files[0]
        
        transcriber = StreamingTranscriber(model_name="tiny")
        detector = InsightDetector(use_llm=False)
        
        chunk_count = 0
        insight_count = 0
        
        async for chunk in transcriber.process_audio(audio_file, chunk_duration=5.0):
            chunk_count += 1
            result = await detector.analyze(chunk)
            insight_count += len(result.insights)
        
        assert chunk_count > 0, "Should process at least one chunk"
        print(f"Processed {chunk_count} chunks, detected {insight_count} insights")
