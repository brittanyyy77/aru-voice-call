# aru-voice-call

Real-time voice call for AI companions — talk to your AI like a phone call.

## Features

- 🎙️ Real-time voice transcription (cloud or local)
- 🎭 Emotion detection from voice (pitch, speed, volume, pauses)
- 🔊 AI voice reply via ElevenLabs or MiniMax TTS
- 📱 Works on mobile browsers, no app needed
- 🐳 One-command Docker deployment
- 🌏 China-friendly: supports Aliyun, Volcengine, Baidu STT + MiniMax TTS
- 🔌 Optional MCP server for Aru integration

## How It Works

The entire call loop runs autonomously on your VPS:

```
You speak → Browser captures audio
         → WebSocket streams to VPS
         → STT transcribes (cloud API or local whisper)
         → librosa analyzes emotion
         → LLM generates reply (with emotion context)
         → TTS synthesizes voice (ElevenLabs or MiniMax)
         → Audio streams back to your browser
         → You hear the reply
```

**No MCP needed for the call itself.** Fully real-time, no middleware delay.

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

Go to `https://your-vps-ip:8443` on your phone.

## Providers

### Speech-to-Text (STT)

| Provider | Region | Free Tier | Config |
|----------|--------|-----------|--------|
| Groq | International | Yes | `STT_PROVIDER=groq` |
| Deepgram | International | Yes | `STT_PROVIDER=deepgram` |
| Aliyun | China | Pay-as-you-go | `STT_PROVIDER=aliyun` + `STT_APPKEY` |
| Volcengine | China | Free quota | `STT_PROVIDER=volcengine` + `STT_APPID` |
| Baidu | China | Free quota | `STT_PROVIDER=baidu` |
| Local Whisper | Any | Free (CPU/GPU) | `STT_MODE=local` |

### Text-to-Speech (TTS)

| Provider | Region | Config |
|----------|--------|--------|
| ElevenLabs | International | `TTS_PROVIDER=elevenlabs` |
| MiniMax | China | `TTS_PROVIDER=minimax` |

## Configuration

See `.env.example` for all options. Key settings:

```env
# International setup
STT_PROVIDER=groq
TTS_PROVIDER=elevenlabs

# China setup (no foreign card needed)
STT_PROVIDER=volcengine
TTS_PROVIDER=minimax
```

### SSL Certificate

Mobile browsers require HTTPS for microphone access:

```bash
mkdir certs
# Copy your fullchain.pem and privkey.pem into certs/
```

## Optional: Aru MCP Integration

If you use [Aru](https://aru.app), bind this as an MCP server:
- URL: `https://your-vps:8443/mcp`
- Transport: `streamable-http`

MCP tools:
- `get_call_status` — Check active/completed calls
- `get_call_transcript` — Get full transcript with emotions after call
- `push_message` — Say something into an active call
- `get_live_emotion` — Check user's mood during call

## Tech Stack

- **Backend**: Python, FastAPI, WebSocket
- **STT**: Groq / Deepgram / Aliyun / Volcengine / Baidu / local whisper
- **Emotion**: librosa audio feature extraction
- **TTS**: ElevenLabs / MiniMax streaming
- **Frontend**: Vanilla HTML/JS, Web Audio API
- **Deploy**: Docker Compose

## License

MIT
