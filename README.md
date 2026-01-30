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
- Console output, or
- Backend endpoint (SSE, WebSockets, async generators)

---

## 📁 Project Structure

```
voice_ai_assignment/
├── src/
│   ├── __init__.py
│   ├── transcription/          # Audio → Text pipeline
│   │   ├── __init__.py
│   │   └── transcriber.py      # Implement streaming transcription
│   ├── insights/               # Real-time insight detection
│   │   ├── __init__.py
│   │   └── detector.py         # Implement insight extraction
│   ├── streaming/              # Output streaming mechanisms
│   │   ├── __init__.py
│   │   └── streamer.py         # Implement streaming output
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       └── audio_utils.py      # Audio processing helpers
├── data/
│   └── samples/                # Place sample audio files here
├── tests/
│   └── __init__.py
├── main.py                     # Main entry point
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice_ai_assignment
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys if needed
```

### Running the Application

```bash
python main.py --audio data/samples/your_audio.wav
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
- [Python SSE](https://pypi.org/project/sse-starlette/)
- [WebSockets in Python](https://websockets.readthedocs.io/)

---

Good luck! 🚀
