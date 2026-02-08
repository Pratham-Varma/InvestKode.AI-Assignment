"""
FastAPI Application for Real-Time Concall Transcription & Insight Streaming

This module provides REST API and SSE endpoints for:
- Uploading and processing audio files
- Streaming transcription results
- Streaming insights in real-time
"""

import asyncio
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import (
    HealthResponse,
    ProcessingStatus,
    TranscriptResponse,
    InsightResponse,
)

# Create FastAPI app
app = FastAPI(
    title="Concall Transcription & Insight Streaming",
    description="Real-time Indian conference call transcription and insight extraction",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for processing status (use Redis in production)
processing_jobs: dict[str, ProcessingStatus] = {}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an audio file for processing.
    
    Returns a job_id that can be used to stream results.
    """
    # Validate file type by extension (content_type is unreliable)
    allowed_extensions = [".wav", ".mp3", ".m4a", ".flac", ".mp4"]
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed extensions: {allowed_extensions}"
        )
    
    # Save uploaded file
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    job_id = str(uuid.uuid4())
    file_path = upload_dir / f"{job_id}_{file.filename}"
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Initialize job status
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="queued",
        file_path=str(file_path),
        progress=0.0,
    )
    
    return {"job_id": job_id, "status": "queued", "message": "File uploaded successfully"}


@app.get("/stream/{job_id}")
async def stream_results(job_id: str):
    """
    Stream transcription and insights via Server-Sent Events (SSE).
    
    Subscribe to this endpoint to receive real-time updates as the audio is processed.
    
    Event types:
    - transcript: New transcript chunk
    - insight: New insight detected
    - summary: Rolling summary update
    - complete: Processing finished
    - error: Processing error
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events for the job."""
        from src.transcription.transcriber import StreamingTranscriber
        from src.insights.detector import InsightDetector
        from src.utils.audio_utils import get_audio_duration
        
        job = processing_jobs[job_id]
        
        try:
            # Update status to processing
            job.status = "processing"
            yield {
                "event": "status",
                "data": json.dumps({"status": "processing", "message": "Starting transcription..."})
            }
            
            # Initialize pipeline
            transcriber = StreamingTranscriber(model_name=os.getenv("WHISPER_MODEL", "base"))
            detector = InsightDetector(use_llm=True)
            
            chunk_duration = float(os.getenv("CHUNK_DURATION_SECONDS", "5"))
            
            # Get total duration for progress tracking
            try:
                total_duration = get_audio_duration(job.file_path)
            except:
                total_duration = None
            
            # Process audio
            chunk_count = 0
            async for chunk in transcriber.process_audio(job.file_path, chunk_duration):
                chunk_count += 1
                
                # Send transcript event
                yield {
                    "event": "transcript",
                    "data": json.dumps({
                        "text": chunk.text,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "confidence": chunk.confidence,
                    })
                }
                
                # Analyze for insights
                result = await detector.analyze(chunk)
                
                # Send insight event if any insights found
                if result.insights:
                    for insight in result.insights:
                        yield {
                            "event": "insight",
                            "data": json.dumps({
                                "type": insight.type.value,
                                "text": insight.text,
                                "sentiment": insight.sentiment.value,
                                "timestamp": insight.timestamp,
                            })
                        }
                
                # Send rolling summary
                if result.rolling_summary:
                    yield {
                        "event": "summary",
                        "data": json.dumps({
                            "summary": result.rolling_summary
                        })
                    }
                
                # Update progress
                if total_duration:
                    job.progress = min((chunk.end_time / total_duration) * 100, 99)
            
            # Generate final summary
            final_summary = detector.get_final_summary()
            all_insights = detector.get_key_insights()
            
            # Save insights to JSON file (matching CLI behavior)
            output_file = None
            if all_insights:
                output_dir = Path("data/outputs")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filename: <audio_filename>-insights-<uuid>.json
                audio_filename = Path(job.file_path).name
                audio_base = Path(audio_filename).stem
                unique_id = str(uuid.uuid4())[:8]
                output_file = output_dir / f"{audio_base}-insights-{unique_id}.json"
                
                # Group insights by type
                insights_by_type = defaultdict(list)
                for insight in all_insights:
                    insights_by_type[insight.type].append(insight)
                
                # Convert insights to dict
                insights_data = {
                    "audio_file": audio_filename,
                    "summary": final_summary,
                    "total_insights": len(all_insights),
                    "insights": [
                        {
                            "type": insight.type.value,
                            "text": insight.text,
                            "sentiment": insight.sentiment.value,
                            "timestamp": insight.timestamp,
                            "confidence": insight.confidence if insight.confidence else 1.0,
                        }
                        for insight in all_insights
                    ],
                    "insights_by_type": {
                        type_name.value: len(type_insights)
                        for type_name, type_insights in insights_by_type.items()
                    }
                }
                
                with open(output_file, 'w') as f:
                    json.dump(insights_data, f, indent=2)
            
            # Send completion
            job.status = "completed"
            job.progress = 100
            yield {
                "event": "complete",
                "data": json.dumps({
                    "status": "completed",
                    "final_summary": final_summary,
                    "total_insights": len(all_insights),
                    "duration_processed": chunk.end_time if chunk_count > 0 else 0,
                    "output_file": str(output_file) if output_file else None,
                })
            }
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "status": "failed"
                })
            }
    
    return EventSourceResponse(event_generator())


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the current status of a processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
    }


@app.delete("/job/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    job.status = "cancelled"
    
    # Clean up file if exists
    if job.file_path and Path(job.file_path).exists():
        Path(job.file_path).unlink()
    
    del processing_jobs[job_id]
    
    return {"message": "Job cancelled successfully"}


# CLI mode - can also run from command line
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
