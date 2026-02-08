#!/usr/bin/env python3
"""
Real-Time Indian Concall Transcription & Insight Streaming

CLI entry point for the application.
Supports both CLI processing and starting the FastAPI server.
"""

import asyncio
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

# CLI app
cli = typer.Typer(
    name="concall",
    help="Real-Time Concall Transcription & Insight Streaming",
    add_completion=False,
)
console = Console()


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable hot reload"),
):
    """Start the FastAPI server."""
    import uvicorn
    
    console.print(f"[bold green]Starting server on {host}:{port}[/bold green]")
    console.print("[dim]API docs available at /docs[/dim]")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
def process(
    audio: Path = typer.Argument(..., help="Path to the audio file to process"),
    chunk_duration: float = typer.Option(
        float(os.getenv("CHUNK_DURATION_SECONDS", "5")),
        "--chunk-duration", "-c",
        help="Duration of each audio chunk in seconds"
    ),
    output: str = typer.Option(
        "console",
        "--output", "-o",
        help="Output method: console, json"
    ),
):
    """
    Process an audio file locally (CLI mode).
    
    This runs the transcription and insight pipeline directly,
    outputting results to the console.
    """
    if not audio.exists():
        console.print(f"[bold red]Error:[/bold red] Audio file not found: {audio}")
        raise typer.Exit(1)
    
    console.print("[bold blue]=" * 60 + "[/bold blue]")
    console.print("[bold]Real-Time Concall Transcription & Insight Streaming[/bold]")
    console.print("[bold blue]=" * 60 + "[/bold blue]")
    console.print(f"[dim]Audio file:[/dim] {audio}")
    console.print(f"[dim]Chunk duration:[/dim] {chunk_duration}s")
    console.print(f"[dim]Output method:[/dim] {output}")
    console.print("[bold blue]=" * 60 + "[/bold blue]\n")
    
    # Run the async processing
    asyncio.run(_process_audio(audio, chunk_duration, output))


async def _process_audio(audio_path: Path, chunk_duration: float, output: str):
    """Process audio file asynchronously."""
    from src.transcription.transcriber import StreamingTranscriber
    from src.insights.detector import InsightDetector
    from src.streaming.streamer import ConsoleStreamer
    from src.utils.device_utils import log_device_info
    
    # Log device information
    log_device_info()
    
    # Initialize components
    console.print("[bold green]Initializing pipeline...[/bold green]")
    transcriber = StreamingTranscriber(model_name=os.getenv("WHISPER_MODEL", "base"))
    detector = InsightDetector(use_llm=True)
    streamer = ConsoleStreamer()
    
    console.print("[bold green]Processing audio...[/bold green]\n")
    
    try:
        # Process audio in chunks
        async for chunk in transcriber.process_audio(audio_path, chunk_duration):
            # Detect insights
            result = await detector.analyze(chunk)
            
            # Stream results
            await streamer.stream(chunk, result)
        
        # Final summary
        console.print("\n")
        final_summary = detector.get_final_summary()
        all_insights = detector.get_key_insights()
        await streamer.stream_final_insights(all_insights, final_summary, audio_filename=audio_path)
        
        console.print(f"\n[bold green]✓ Processing complete![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error during processing: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(1)
    finally:
        await streamer.close()


@cli.command()
def version():
    """Show version information."""
    console.print("[bold]Concall Transcription[/bold] v0.1.0")


if __name__ == "__main__":
    cli()
