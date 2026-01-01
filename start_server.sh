#!/usr/bin/env bash
#
# Dodger Server Startup Script
# ============================
# This script automates the process of:
# 1. Installing cloudflared (if not present)
# 2. Installing Python dependencies
# 3. Starting the Dodger server
# 4. Creating a Cloudflare tunnel
# 5. Displaying connection credentials
#
# Usage: ./start_server.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
info() { echo -e "${BLUE}ℹ${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }

# Header
echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    🚀 Dodger Server                        ║"
echo "║              Local LLM Backend for Mobile                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    elif [[ -f /etc/fedora-release ]]; then
        echo "fedora"
    elif [[ -f /etc/arch-release ]]; then
        echo "arch"
    else
        echo "linux"
    fi
}

OS=$(detect_os)
info "Detected OS: $OS"

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
        
        if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 11 ]]; then
            success "Python $PY_VERSION found"
            return 0
        else
            error "Python 3.11+ required (found $PY_VERSION)"
        fi
    else
        error "Python 3 not found. Please install Python 3.11 or newer."
    fi
}

# Install cloudflared
install_cloudflared() {
    if command -v cloudflared &> /dev/null; then
        success "cloudflared already installed"
        return 0
    fi
    
    info "Installing cloudflared..."
    
    case "$OS" in
        macos)
            if command -v brew &> /dev/null; then
                brew install cloudflared
            else
                error "Homebrew not found. Install it from https://brew.sh or install cloudflared manually."
            fi
            ;;
        debian)
            # Official Cloudflare installation for Debian/Ubuntu
            curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
            sudo dpkg -i cloudflared.deb
            rm cloudflared.deb
            ;;
        fedora)
            sudo dnf install -y cloudflared || {
                # Fallback to binary install
                curl -L --output /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
                sudo install -m 755 /tmp/cloudflared /usr/local/bin/cloudflared
            }
            ;;
        arch)
            if command -v yay &> /dev/null; then
                yay -S --noconfirm cloudflared
            elif command -v paru &> /dev/null; then
                paru -S --noconfirm cloudflared
            else
                # Fallback to binary install
                curl -L --output /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
                sudo install -m 755 /tmp/cloudflared /usr/local/bin/cloudflared
            fi
            ;;
        *)
            # Generic Linux binary install
            curl -L --output /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
            sudo install -m 755 /tmp/cloudflared /usr/local/bin/cloudflared
            ;;
    esac
    
    if command -v cloudflared &> /dev/null; then
        success "cloudflared installed successfully"
    else
        error "Failed to install cloudflared"
    fi
}

# Install Python dependencies
install_dependencies() {
    info "Installing Python dependencies..."
    
    # Check if we're in a virtual environment or use pip
    if [[ -n "$VIRTUAL_ENV" ]]; then
        pip install -e . --quiet
    elif command -v uv &> /dev/null; then
        uv pip install -e . --quiet
    else
        python3 -m pip install -e . --quiet --user
    fi
    
    success "Dependencies installed"
}

# Get or generate API token
get_api_token() {
    CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/dodger"
    CONFIG_FILE="$CONFIG_DIR/config.json"
    
    if [[ -f "$CONFIG_FILE" ]]; then
        # Try to extract token from existing config
        TOKEN=$(python3 -c "
import json
try:
    with open('$CONFIG_FILE') as f:
        config = json.load(f)
        token = config.get('server', {}).get('api_token', '')
        print(token if token else '')
except:
    print('')
" 2>/dev/null)
        
        if [[ -n "$TOKEN" ]]; then
            echo "$TOKEN"
            return 0
        fi
    fi
    
    # Generate new token
    TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    echo "$TOKEN"
}

# Start the server
start_server() {
    info "Starting Dodger server..."
    
    # Kill any existing server
    pkill -f "dodger-server" 2>/dev/null || true
    
    # Start server in background
    dodger-server --host 0.0.0.0 --port 8000 &> /tmp/dodger-server.log &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 3
    
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        success "Server started (PID: $SERVER_PID)"
    else
        error "Failed to start server. Check /tmp/dodger-server.log for details."
    fi
}

# Start cloudflare tunnel
start_tunnel() {
    info "Creating Cloudflare tunnel..."
    
    # Kill any existing tunnel
    pkill -f "cloudflared tunnel" 2>/dev/null || true
    
    # Start tunnel and capture output
    TUNNEL_LOG="/tmp/cloudflared.log"
    cloudflared tunnel --url http://localhost:8000 2>&1 | tee "$TUNNEL_LOG" &
    TUNNEL_PID=$!
    
    # Wait for tunnel URL to appear
    info "Waiting for tunnel to establish..."
    
    TUNNEL_URL=""
    for i in {1..30}; do
        sleep 1
        TUNNEL_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
        if [[ -n "$TUNNEL_URL" ]]; then
            break
        fi
    done
    
    if [[ -z "$TUNNEL_URL" ]]; then
        error "Failed to establish tunnel. Check /tmp/cloudflared.log for details."
    fi
    
    success "Tunnel established!"
}

# Display connection info
display_info() {
    API_TOKEN=$(get_api_token)
    
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   🎉 Server Ready!                         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Connect to your server from the web app:${NC}"
    echo ""
    echo -e "  ${CYAN}Server URL:${NC}  $TUNNEL_URL"
    echo -e "  ${CYAN}API Token:${NC}   $API_TOKEN"
    echo ""
    echo -e "${YELLOW}Copy these values into the Dodger web app settings.${NC}"
    echo ""
    echo -e "${BOLD}Web App:${NC} https://recursiveintell.com"
    echo ""
    echo -e "Press ${RED}Ctrl+C${NC} to stop the server and tunnel."
    echo ""
    
    # Keep the script running to maintain tunnel
    wait $TUNNEL_PID
}

# Cleanup on exit
cleanup() {
    echo ""
    info "Shutting down..."
    pkill -f "dodger-server" 2>/dev/null || true
    pkill -f "cloudflared tunnel" 2>/dev/null || true
    success "Goodbye!"
}

trap cleanup EXIT INT TERM

# Main execution
main() {
    check_python
    install_cloudflared
    install_dependencies
    start_server
    start_tunnel
    display_info
}

main "$@"
