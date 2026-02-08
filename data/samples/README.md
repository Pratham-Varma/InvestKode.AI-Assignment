# Sample Audio Files

Place your sample conference call audio files here for testing.

## Supported Formats
- WAV (recommended)
- MP3
- M4A
- FLAC

## Where to Find Sample Audio

### Option 1: Record Your Own
Record a 30-60 second audio clip simulating a conference call:
- "Good morning everyone, welcome to our Q3 earnings call..."
- "Revenue this quarter grew by 15% year over year to 500 crores..."
- "We are targeting 20% growth next quarter..."

### Option 2: Download from Free Sources
- [Freesound.org](https://freesound.org/) - Search for "speech" or "conference"
- Use text-to-speech services to generate audio

### Option 3: YouTube
Download audio from Indian earnings call videos (audio only, respecting copyright)

## Testing

Once you have audio files here, test with:

```bash
# CLI mode
uv run python main.py process data/samples/your_audio.wav

# API mode
curl -X POST http://localhost:8000/upload -F "file=@data/samples/your_audio.wav"
```

## Note

The `data/` directory is gitignored, so your audio files won't be committed to the repository.
