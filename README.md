# 🚀 Local LLM Backend

A Python/FastAPI backend for Dodger-compatible clients, supporting local model services and configured cloud providers. This repository contains the backend, not a bundled web client.

## ✨ Features

- **Multi-Provider Support**: Connect to Ollama, LMStudio, OpenAI, Anthropic, Google AI, and OpenRouter
- **Optional remote access**: The convenience launcher creates a Cloudflare tunnel and exposes the API remotely
- **Web Search**: Built-in DuckDuckGo search integration
- **Streaming**: Real-time response streaming via WebSocket
- **Persistent configuration**: Provider settings, API token and network binding are stored in the user configuration directory

## 🏃 Quick Start

### Convenience launcher (installs software and exposes a tunnel)

```bash
git clone https://github.com/RecursiveIntell/local-llm-backend.git
cd local-llm-backend
./start_server.sh
```

Review `start_server.sh` before using it. It may install system software with sudo, starts a public tunnel, and uses broad `pkill` patterns for `dodger-server` and `cloudflared tunnel`, which can affect other matching processes. Use the manual loopback setup below when remote access is not intended.

The script will:
1. ✅ Check for Python 3.11+
2. ✅ Install `cloudflared` if needed
3. ✅ Install dependencies
4. ✅ Start the server
5. ✅ Create an HTTPS tunnel (transport encryption does not establish application security)
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

# Set the saved binding explicitly before starting
python -c "from dodger.config import config_manager; config_manager.load(); config_manager.settings.server.host = '127.0.0.1'; config_manager.settings.server.port = 8000; config_manager.save()"
dodger-server
```

The current `dodger-server` entry point does not parse `--host` or `--port`; it reads `server.host` and `server.port` from configuration. Both the model defaults and the launcher use `0.0.0.0`, so this is not loopback-only by default. The manual command above deliberately saves a loopback binding.

## Connecting a client

Use a compatible client or the REST/WebSocket API below. Historical launcher text points to `recursiveintell.com`, but that link does not establish that a compatible hosted client is available. Do not paste a server token into an unverified website.

For protected REST routes, send `Authorization: Bearer <your-token>`. The current WebSocket route reads a `token` query parameter; avoid sharing those URLs or logging them. The initial token is generated and printed by the server when none exists; save it securely. Configuration and the token are stored under the platform's user configuration directory (typically `~/.config/dodger/config.json` on Linux).

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

Configure provider credentials through a compatible client or the authenticated provider configuration routes. Cloud requests send the selected content to that provider and may incur its usage charges:

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
- **Binding**: Default `0.0.0.0` listens on every interface; configure loopback explicitly for local-only access
- **Deployment**: A tunnel URL is a public exposure choice. Protect the token/configuration, review CORS, and test authentication before sharing access. No production-security certification is claimed

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Server health check |
| `/models` | GET | List available models (Bearer token) |
| `/conversations` | GET | List stored conversations (Bearer token) |
| `/providers` | GET | Inspect provider configuration (Bearer token) |
| `/providers/{provider_name}` | PUT | Change provider settings (Bearer token) |
| `/config/active` | GET / PUT | Read or select active provider/model (Bearer token) |
| `/token/generate` | POST | Rotate token using an existing valid token |
| `/chat` | POST | Send chat message |
| `/chat/stream` | WebSocket | Stream responses |
| `/search` | POST | Web search |

## Source checks

```bash
python -m compileall -q src/dodger
```

This checks syntax only. Test a loopback client, authentication failures, provider errors and stream termination before exposing the service. Live model/cloud/tunnel behavior was not validated by this documentation update.

## 📄 License

`pyproject.toml` declares MIT, but this checkout does not include a standalone `LICENSE` file.

---

**Need help?** Open an issue on [GitHub](https://github.com/RecursiveIntell/local-llm-backend/issues)
