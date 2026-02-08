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
    
    async def stream_final_insights(self, insights: list, summary: str, audio_filename: str = None) -> None:
        """Stream final insights with details and save to JSON file."""
        import json
        import uuid
        from datetime import datetime
        from pathlib import Path
        
        print(f"\n{'=' * 70}")
        print("📋 FINAL SUMMARY & INSIGHTS")
        print(f"{'=' * 70}")
        print(f"\n{summary}\n")
        
        if insights:
            print(f"\n{'─' * 70}")
            print(f"📊 DETAILED INSIGHTS ({len(insights)} total):\n")
            
            # Group insights by type
            from collections import defaultdict
            insights_by_type = defaultdict(list)
            for insight in insights:
                insights_by_type[insight.type].append(insight)
            
            # Display each type
            for insight_type, type_insights in insights_by_type.items():
                print(f"\n  [{insight_type.value.upper()}] ({len(type_insights)} mentions):")
                for i, insight in enumerate(type_insights[:5], 1):  # Show first 5 of each type
                    sentiment_emoji = "📈" if insight.sentiment.value == "positive" else "📉" if insight.sentiment.value == "negative" else "➡️"
                    # Show full text, but limit to reasonable length for console
                    display_text = insight.text if len(insight.text) <= 150 else insight.text[:147] + "..."
                    print(f"    {i}. {sentiment_emoji} {display_text}")
                    confidence_pct = f"{insight.confidence*100:.0f}%" if insight.confidence else "N/A"
                    print(f"       └─ Time: {insight.timestamp:.1f}s | Sentiment: {insight.sentiment.value} | Confidence: {confidence_pct}")
                
                if len(type_insights) > 5:
                    print(f"    ... and {len(type_insights) - 5} more")
            
            print(f"\n{'─'  * 70}")
            
            # Save to JSON file
            output_dir = Path("data/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename: <audio_filename>-insights-<uuid>.json
            if audio_filename:
                # Extract basename without extension
                audio_base = Path(audio_filename).stem
                unique_id = str(uuid.uuid4())[:8]
                output_file = output_dir / f"{audio_base}-insights-{unique_id}.json"
            else:
                # Fallback to timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"insights_{timestamp}.json"
            
            # Convert insights to dict
            insights_data = {
                "audio_file": str(audio_filename) if audio_filename else "unknown",
                "summary": summary,
                "total_insights": len(insights),
                "insights": [
                    {
                        "type": insight.type.value,
                        "text": insight.text,  # Full text, no truncation
                        "sentiment": insight.sentiment.value,
                        "timestamp": insight.timestamp,
                        "confidence": insight.confidence if insight.confidence else 1.0,
                    }
                    for insight in insights
                ],
                "insights_by_type": {
                    type_name.value: len(type_insights)
                    for type_name, type_insights in insights_by_type.items()
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(insights_data, f, indent=2)
            
            print(f"\n💾 Insights saved to: {output_file}")
        else:
            print("\nNo insights detected.")
        
        print(f"{'=' * 70}\n")
    
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
