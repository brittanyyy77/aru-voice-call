# aru-voice-call

Real-time voice call for AI companions — talk to your AI like a phone call.

## Features

- 🎙️ Real-time voice transcription (faster-whisper)
- 🎭 Emotion detection from voice (pitch, speed, volume, pauses)
- 🔊 AI voice reply via ElevenLabs TTS
- 🔌 MCP server for Aru integration — your AI keeps its personality, memory, and world books
- 📱 Works on mobile browsers, no app needed
- 🐳 One-command Docker deployment

## Architecture

```
Browser (mic) → WebSocket → VPS Backend
                                ├── faster-whisper (transcription)
                                ├── emotion analysis (librosa)
                                ├── LLM API (your model)
                                └── ElevenLabs TTS → WebSocket → Browser (speaker)
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/brittanyyy77/aru-voice-call.git
cd aru-voice-call
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with Docker

```bash
docker compose up -d
```

### 3. Open in browser

Go to `https://your-vps-ip:8443` on your phone or computer.

Press the call button and start talking.

## Configuration

Edit `.env` file:

```env
# Required
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id

# LLM (for standalone mode, not needed for Aru MCP)
LLM_API_KEY=your_llm_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

# Optional
WHISPER_MODEL=large-v3
LANGUAGE=zh
PORT=8443
SSL_CERT=/certs/fullchain.pem
SSL_KEY=/certs/privkey.pem
```

## Aru MCP Integration

This server exposes MCP tools that Aru can bind to:

- `start_call` — Initialize a voice call session
- `end_call` — End the current call
- `get_transcript` — Get real-time transcription with emotion tags
- `send_voice_reply` — Send text to be spoken back to the user

In Aru, add this as an MCP server with your VPS URL.

## Tech Stack

- **Backend**: Python, FastAPI, WebSocket
- **Transcription**: faster-whisper (local, GPU optional)
- **Emotion**: librosa audio feature extraction
- **TTS**: ElevenLabs Streaming API
- **Frontend**: Vanilla HTML/JS, Web Audio API
- **Deploy**: Docker Compose

## License

MIT
