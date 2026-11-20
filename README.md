# aru-voice-call

Real-time voice call for AI companions — talk to your AI like a phone call.

## Features

- 🎙️ Real-time voice transcription (faster-whisper)
- 🎭 Emotion detection from voice (pitch, speed, volume, pauses)
- 🔊 AI voice reply via ElevenLabs TTS
- 📱 Works on mobile browsers, no app needed
- 🐳 One-command Docker deployment
- 🔌 Optional MCP server for Aru — post-call transcript sync & push messages

## How It Works

The entire call loop runs autonomously on your VPS:

```
You speak → Browser captures audio
         → WebSocket streams to VPS
         → faster-whisper transcribes in real-time
         → librosa analyzes emotion (pitch, energy, speed)
         → LLM generates reply (with emotion context)
         → ElevenLabs synthesizes voice
         → Audio streams back to your browser
         → You hear the reply
```

**No MCP needed for the call itself.** The call is fully real-time, no middleware delay.

### Optional: Aru MCP Integration

If you use [Aru](https://aru.app), bind this as an MCP server to:

- **Post-call sync**: After hanging up, Aru can retrieve the full transcript with emotion tags and save it to memory. Your AI remembers what you talked about on the phone.
- **Push messages**: Aru can push a message into an active call (e.g. proactive check-in), which gets synthesized and played to you.
- **Live emotion**: Aru can check your current emotional state during a call.

This means your AI's personality, memory, and world books all stay in Aru where they belong. The call server just handles voice I/O.

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

# LLM
LLM_API_KEY=your_llm_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
SYSTEM_PROMPT=You are a caring AI companion.

# Optional
WHISPER_MODEL=large-v3
LANGUAGE=zh
PORT=8443
SSL_CERT=/certs/fullchain.pem
SSL_KEY=/certs/privkey.pem
MCP_ENABLED=true
```

### SSL Certificate

Mobile browsers require HTTPS for microphone access. Place your cert files in `./certs/` or use Let's Encrypt:

```bash
mkdir certs
# Copy your fullchain.pem and privkey.pem into certs/
```

## Aru MCP Setup

In Aru settings, add MCP server:
- URL: `https://your-vps:8443/mcp`
- Transport: `streamable-http`

Available MCP tools:
- `get_call_status` — Check active/completed calls
- `get_call_transcript` — Get full transcript with emotions after call ends
- `push_message` — Say something into an active call
- `get_live_emotion` — Check user's current mood during call

## Tech Stack

- **Backend**: Python, FastAPI, WebSocket
- **Transcription**: faster-whisper (local, GPU optional, CPU fallback)
- **Emotion**: librosa audio feature extraction
- **TTS**: ElevenLabs Streaming API
- **Frontend**: Vanilla HTML/JS, Web Audio API
- **Deploy**: Docker Compose

## License

MIT
