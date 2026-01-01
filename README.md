# 🚀 Local LLM Backend

A lightweight backend server that lets you access your local LLMs (Ollama, LMStudio) or cloud APIs (OpenAI, Anthropic, Google) from anywhere via the [Dodger Web App](https://recursiveintell.com).

## ✨ Features

- **Multi-Provider Support**: Connect to Ollama, LMStudio, OpenAI, Anthropic, Google AI, and OpenRouter
- **Secure Tunneling**: Auto-creates a Cloudflare tunnel for remote access
- **Web Search**: Built-in DuckDuckGo search integration
- **Streaming**: Real-time response streaming via WebSocket
- **Zero Config**: Works out of the box with sensible defaults

## 🏃 Quick Start

### One-Command Install (Linux/macOS)

```bash
git clone https://github.com/RecursiveIntell/local-llm-backend.git
cd local-llm-backend
./start_server.sh
```

The script will:
1. ✅ Check for Python 3.11+
2. ✅ Install `cloudflared` if needed
3. ✅ Install dependencies
4. ✅ Start the server
5. ✅ Create a secure tunnel
6. ✅ Display your **connection credentials**

### Manual Setup

```bash
# Clone the repo
git clone https://github.com/RecursiveIntell/local-llm-backend.git
cd local-llm-backend

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Start the server
dodger-server --host 0.0.0.0 --port 8000
```

## 🔗 Connecting to the Web App

1. Run the server using the quick start above
2. Copy the **Tunnel URL** and **API Token** displayed
3. Go to [recursiveintell.com](https://recursiveintell.com)
4. Click ⚙️ **Settings** → **Server Connection**
5. Paste your credentials and click **Connect**

## 🤖 Setting Up Local LLMs

### Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Ollama runs on localhost:11434 by default
```

### LMStudio

1. Download [LMStudio](https://lmstudio.ai/)
2. Download a model (e.g., Qwen, Mistral)
3. Start the local server (runs on localhost:1234)

## ☁️ Cloud Providers

Configure API keys in the web app's Settings:

| Provider | Get API Key |
|----------|-------------|
| OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| Google AI | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| OpenRouter | [openrouter.ai](https://openrouter.ai/keys) |

## 📁 Project Structure

```
local-llm-backend/
├── start_server.sh     # One-click setup script
├── pyproject.toml      # Python dependencies
└── src/dodger/
    ├── config.py       # Configuration management
    ├── core/           # Core utilities
    ├── providers/      # LLM provider implementations
    └── server/         # FastAPI server
```

## 🔒 Security

- **API Token**: Auto-generated 256-bit secure token
- **HTTPS**: Cloudflare tunnel provides automatic HTTPS
- **Local Only**: Server binds to localhost by default

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Server health check |
| `/models` | GET | List available models |
| `/chat` | POST | Send chat message |
| `/chat/stream` | WebSocket | Stream responses |
| `/search` | POST | Web search |

## 📄 License

MIT License - Use freely!

---

**Need help?** Open an issue on [GitHub](https://github.com/RecursiveIntell/local-llm-backend/issues)
