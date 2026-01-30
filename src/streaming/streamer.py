"""
Streaming Output Module

Implement your output streaming mechanism here.

Requirements:
- Stream results in real-time as they are generated
- Support at least one of: console, SSE, or WebSocket

Suggested Approaches:
- Console: Simple print with rich formatting
- SSE: Use sse-starlette with FastAPI
- WebSocket: Use websockets or FastAPI WebSocket
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Optional

from src.insights.detector import InsightResult
from src.transcription.transcriber import TranscriptChunk


class BaseStreamer(ABC):
    """Abstract base class for streaming output."""
    
    @abstractmethod
    async def stream(
        self, 
        chunk: TranscriptChunk, 
        insights: Optional[InsightResult] = None
    ) -> None:
        """
        Stream a transcript chunk and its insights.
        
        Args:
            chunk: The transcribed audio chunk
            insights: Optional insights extracted from the chunk
        """
        pass
    
    @abstractmethod
    async def stream_summary(self, summary: str) -> None:
        """Stream a summary update."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class ConsoleStreamer(BaseStreamer):
    """
    Stream output to console with formatting.
    
    This is the simplest implementation. Start here and optionally
    add SSE or WebSocket support.
    """
    
    def __init__(self, use_rich: bool = True):
        """
        Initialize console streamer.
        
        Args:
            use_rich: Whether to use rich library for formatting
        """
        self.use_rich = use_rich
        
        # Optional: Use rich for better formatting
        # try:
        #     from rich.console import Console
        #     from rich.panel import Panel
        #     self.console = Console()
        # except ImportError:
        #     self.use_rich = False
    
    async def stream(
        self, 
        chunk: TranscriptChunk, 
        insights: Optional[InsightResult] = None
    ) -> None:
        """Stream transcript and insights to console."""
        # TODO: Implement console streaming
        # 
        # Example output format:
        # 
        # ─────────────────────────────────────────
        # [00:05 - 00:10] Transcript:
        # "The revenue for this quarter was 500 crores..."
        # 
        # 📊 Insights:
        #   • [REVENUE] 500 crores mentioned
        #   • [GROWTH] Positive growth signal
        # 
        # 📝 Rolling Summary:
        # Company reported Q3 revenue of 500 crores...
        # ─────────────────────────────────────────
        
        timestamp = f"[{chunk.start_time:.0f}s - {chunk.end_time:.0f}s]"
        print(f"\n{'─' * 50}")
        print(f"{timestamp} Transcript:")
        print(f'"{chunk.text}"')
        
        if insights:
            if insights.insights:
                print("\n📊 Insights:")
                for insight in insights.insights:
                    print(f"  • [{insight.type.value.upper()}] {insight.text}")
            
            if insights.rolling_summary:
                print(f"\n📝 Summary: {insights.rolling_summary}")
        
        print(f"{'─' * 50}")
    
    async def stream_summary(self, summary: str) -> None:
        """Stream a summary update."""
        print(f"\n{'=' * 50}")
        print("📋 FINAL SUMMARY")
        print(f"{'=' * 50}")
        print(summary)
        print(f"{'=' * 50}\n")
    
    async def close(self) -> None:
        """Clean up (no-op for console)."""
        pass


class SSEStreamer(BaseStreamer):
    """
    Stream output via Server-Sent Events.
    
    TODO: Implement if you want to support web-based clients.
    
    This requires setting up a FastAPI server with SSE endpoints.
    """
    
    def __init__(self, port: int = 8000):
        self.port = port
        # TODO: Initialize FastAPI app and SSE mechanism
        
    async def stream(
        self, 
        chunk: TranscriptChunk, 
        insights: Optional[InsightResult] = None
    ) -> None:
        """Stream via SSE."""
        # TODO: Implement SSE streaming
        # 
        # Example with sse-starlette:
        # 
        # from sse_starlette.sse import EventSourceResponse
        # 
        # async def event_generator():
        #     data = {
        #         "type": "transcript",
        #         "chunk": asdict(chunk),
        #         "insights": asdict(insights) if insights else None,
        #     }
        #     yield {"data": json.dumps(data)}
        
        raise NotImplementedError("TODO: Implement SSE streaming")
    
    async def stream_summary(self, summary: str) -> None:
        """Stream summary via SSE."""
        raise NotImplementedError("TODO: Implement SSE summary streaming")
    
    async def close(self) -> None:
        """Clean up SSE resources."""
        raise NotImplementedError("TODO: Implement SSE cleanup")


class WebSocketStreamer(BaseStreamer):
    """
    Stream output via WebSocket.
    
    TODO: Implement if you want bidirectional communication.
    """
    
    def __init__(self, port: int = 8000):
        self.port = port
        # TODO: Initialize WebSocket server
        
    async def stream(
        self, 
        chunk: TranscriptChunk, 
        insights: Optional[InsightResult] = None
    ) -> None:
        """Stream via WebSocket."""
        raise NotImplementedError("TODO: Implement WebSocket streaming")
    
    async def stream_summary(self, summary: str) -> None:
        """Stream summary via WebSocket."""
        raise NotImplementedError("TODO: Implement WebSocket summary streaming")
    
    async def close(self) -> None:
        """Clean up WebSocket resources."""
        raise NotImplementedError("TODO: Implement WebSocket cleanup")
