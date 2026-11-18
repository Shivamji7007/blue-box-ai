# Blue Box AI - 100% Offline, Screen-Free Pocket Study Tutor

Blue Box AI is a zero-distraction, 100% offline voice AI study companion designed for Indian students.

## Architecture
- **Speech-to-Text:** OpenAI Whisper (base.en via pywhispercpp)
- **Offline Brain:** qwen2.5:1.5b running locally via Ollama (localhost:11434)
- **Local Knowledge (RAG):** Pure-Python offline search across local .txt, .md, and .pdf notes
- **Voice Output:** Offline Text-to-Speech (pyttsx3)

## Quick Start
1. Install Ollama and run: `ollama pull qwen2.5:1.5b`
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python blue_box_ai.py`
