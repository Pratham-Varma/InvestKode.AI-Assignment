# Submission Checklist

## Pre-Submission Verification

### ✅ Code Quality
- [x] All redundant imports removed
- [x] TODO comments removed from implemented code
- [x] Code is clean and readable
- [x] No sensitive data (API keys) in committed files

### ✅ Files Ready
- [x] `requirements.txt` generated from pyproject.toml
- [x] `.gitignore` properly configured
- [x] `README.md` updated with implementation details
- [x] `uv.lock` committed for reproducibility

### 🔍 Testing Checklist

#### CLI Mode Test
```bash
# Test command
cd voice-ai-assignment
uv run python main.py process data/samples/gulf_oil_india_concall.wav

# Expected outputs:
# ✓ Transcription and insights printed to console
# ✓ JSON file saved to data/outputs/gulf_oil_india_concall-insights-<uuid>.json
```

#### API Mode Test
```bash
# Terminal 1: Start server
uv run python main.py serve

#Terminal 2: Upload and stream
JOB_ID=$(curl -X POST http://localhost:8000/upload -F "file=@data/samples/gulf_oil_india_concall.wav" | jq -r '.job_id')
curl -N http://localhost:8000/stream/$JOB_ID

# Expected outputs:
# ✓ SSE events streamed (transcript, insight, summary, complete)
# ✓ JSON file saved to data/outputs/
```

### 📧 Email Submission

**To:** team@investkode.ai  
**Subject:** Assignment Submission – Voice AI Engineer – [Your Name]

**Email Body Template:**

```
Hello,

I am submitting my Voice AI Engineer assignment. Please find the details below:

GitHub Repository: [Your GitHub Repo URL]
Note: Repository is public as requested

Key Features Implemented:
✓ Streaming transcription using Whisper (faster-whisper)
✓ Real-time insight detection with Google Gemini API + rule-based fallback
✓ FastAPI server with Server-Sent Events (SSE)
✓ CLI mode for local processing
✓ JSON output generation in both modes
✓ Sentiment analysis integration
✓ Docker support

Technologies Used:
- ASR: faster-whisper (Whisper model)
- LLM: Google Gemini API (gemini-2.5-flash) with rule-based fallback
- API Framework: FastAPI with sse-starlette
- Package Manager: uv
- Python 3.10+

Testing:
Both CLI and API modes have been tested successfully with the provided sample audio file.

Thank you for your consideration.

Best regards,
[Your Name]
```

### 📦 What to Include in Repository

**Required Files:**
- ✅ All source code in `src/`
- ✅ `main.py` (CLI entrypoint)
- ✅ `requirements.txt`
- ✅ `pyproject.toml`
- ✅ `uv.lock`
- ✅ `README.md` (with implementation details)
- ✅ `.env.example` (template)
- ✅ `Dockerfile` and `docker-compose.yml`
- ✅ `.gitignore`
- ✅ Sample audio file in `data/samples/`

**What NOT to Include:**
- ❌ `.env` (contains API keys)
- ❌ `.venv/` or `venv/` (virtual environment)
- ❌ `__pycache__/` (Python cache)
- ❌ `data/uploads/` (uploaded files)
- ❌ `data/outputs/*.json` (generated outputs - optional to include as examples)
- ❌ Model cache files

### 🚀 Final Steps Before Submission

1. **Clean up generated files:**
   ```bash
   rm -rf data/uploads/*
   # Optional: keep 1-2 example JSON outputs in data/outputs/ to demonstrate
   ```

2. **Verify repository is clean:**
   ```bash
   git status
   # Ensure no sensitive files are tracked
   ```

3. **Test from fresh clone:**
   ```bash
   cd /tmp
   git clone <your-repo-url>
   cd voice-ai-assignment
   uv sync
   uv run python main.py serve
   # Verify it works
   ```

4. **Make repository public** (if not already)

5. **Send email** with GitHub link

---

## Important Notes

- The `.env` file should **NEVER** be committed. Only `.env.example` is committed.
- Keep `uv.lock` committed for reproducible builds
- Sample audio files in `data/samples/` should be small (< 50MB)
- Generated outputs in `data/outputs/` can be gitignored or include 1-2 examples
