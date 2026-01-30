# Real-Time Indian Concall Transcription & Insight Streaming

## 🎯 Assignment Overview

Build a prototype system that processes **Indian earnings / conference calls (concalls)** and generates **live insights while the call is happening**.

**Role:** Voice AI Engineer  
**Time Limit:** 3 Days  
**Expected Effort:** 6–8 hours  

---

## 📋 Problem Statement

Your system should simulate or demonstrate:
- Streaming audio transcription
- Real-time insight detection  
- Live streaming of outputs

---

## 🏗️ What You Need to Build

### 1. Streaming Transcription (Audio → Text)
- Use a short audio clip of an Indian concall (real or simulated)
- Process audio in chunks or near-real-time
- Convert speech to text using:
  - Open-source ASR (e.g., Whisper), or
  - Any speech API (free tier or mocked)

### 2. Real-Time Insight Detection
As transcript chunks arrive, generate:
- Rolling summaries
- Key financial signals (revenue, guidance, risks, outlook, etc.)
- Any insights relevant for equity research

### 3. Streaming Output
Stream results using:
- **FastAPI with SSE** (Server-Sent Events) - Primary method
- Console output for CLI mode

---

## 📁 Project Structure

```
voice_ai_assignment/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py             # API endpoints & SSE streaming
│   │   └── schemas.py          # Pydantic models
│   ├── transcription/          # Audio → Text pipeline
│   │   └── transcriber.py      # Implement streaming transcription
│   ├── insights/               # Real-time insight detection
│   │   └── detector.py         # Implement insight extraction
│   ├── streaming/              # Output streaming mechanisms
│   │   └── streamer.py         # Console/WebSocket streamers
│   └── utils/                  # Shared utilities
│       └── audio_utils.py      # Audio processing helpers
├── data/
│   └── samples/                # Place sample audio files here
├── tests/
├── main.py                     # CLI entry point
├── pyproject.toml              # Dependencies (uv compatible)
├── Dockerfile                  # Production container
├── Dockerfile.dev              # Development container
├── docker-compose.yml          # Container orchestration
├── .env.example
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker (optional)

### Option 1: Local Development with uv (Recommended)

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd voice_ai_assignment

# Install dependencies
uv sync

# Optional: Install ASR and LLM packages
uv sync --extra whisper --extra openai

# Run the FastAPI server
uv run python main.py serve

# Or process a file directly
uv run python main.py process data/samples/your_audio.wav
```

### Option 2: Docker

```bash
# Build and run production container
docker-compose up --build

# Or run development mode with hot reload
docker-compose --profile dev up dev
```

### Option 3: Traditional pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
python main.py serve
```

---

## 🔌 API Endpoints

Once running, the API is available at `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Interactive API documentation |
| `/health` | GET | Health check |
| `/upload` | POST | Upload audio file, returns job_id |
| `/stream/{job_id}` | GET | SSE stream of transcription & insights |
| `/status/{job_id}` | GET | Check job status |
| `/job/{job_id}` | DELETE | Cancel a job |

### SSE Event Types

When subscribing to `/stream/{job_id}`, you'll receive:

```
event: transcript
data: {"text": "...", "start_time": 0, "end_time": 5}

event: insight
data: {"type": "revenue", "text": "...", "sentiment": "positive"}

event: summary
data: {"summary": "Rolling summary..."}

event: complete
data: {"status": "completed", "final_summary": "..."}
```

---

## 🖥️ CLI Usage

```bash
# Start the server
uv run python main.py serve --port 8000 --reload

# Process audio file locally
uv run python main.py process audio.wav --chunk-duration 5

# Show help
uv run python main.py --help
```

---

## ✅ Technical Constraints

- **Primary language:** Python (scripts only)
- No Jupyter or notebook-based solutions
- Code must run in a local IDE
- Use clean, modular code

---

## 📝 Your README Should Explain

When you complete the assignment, update this README to include:

- [ ] What you built
- [ ] High-level architecture
- [ ] How streaming is handled
- [ ] Assumptions and tradeoffs
- [ ] What you would improve with more time

**Optional:** Logs or screenshots demonstrating streaming output

---

## 🎁 Bonus (Optional)

- Speaker diarization (Management vs Analyst)
- Detection of new or changing information
- Sentiment shifts during the call
- Handling Indian accents or Hinglish
- Hindi-to-English translation

---

## 📊 Evaluation Criteria

We'll evaluate:
- Understanding of real-time systems
- Handling of messy, domain-specific audio
- Quality and usefulness of extracted insights
- Code structure, readability, and fundamentals
- Ownership and clarity of reasoning

**What we DON'T expect:**
- Perfect transcription accuracy
- Fully live market or audio integrations
- Production-grade infrastructure

---

## 📬 Submission Instructions

1. Complete your implementation
2. Update this README with your documentation
3. Ensure your repository is public
4. Email your submission to: team@investkode.ai
   - **Subject:** `Assignment Submission – Voice AI Engineer – [Your Name]`
   - Include: Link to your GitHub repository

---

## 📚 Resources

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [FastAPI SSE](https://github.com/sysid/sse-starlette)
- [uv - Fast Python Package Manager](https://github.com/astral-sh/uv)

---

Good luck! 🚀
