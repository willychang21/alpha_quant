#!/bin/bash
# ==============================================================================
# DCA Quant - Frontend Launcher
# ==============================================================================

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log_info() { echo -e "\033[0;90m[$(timestamp)]\033[0m ${BLUE}FRONTEND${NC} $1"; }
log_success() { echo -e "\033[0;90m[$(timestamp)]\033[0m ${GREEN}FRONTEND${NC} $1"; }
log_warn() { echo -e "\033[0;90m[$(timestamp)]\033[0m ${YELLOW}FRONTEND${NC} $1"; }
log_error() { echo -e "\033[0;90m[$(timestamp)]\033[0m ${RED}FRONTEND${NC} $1" >&2; }

# Environment Check
check_env() {
    if [[ ! -d "${FRONTEND_DIR}" ]]; then
        log_error "Directory not found: ${FRONTEND_DIR}"
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        log_error "npm not found. Please install Node.js."
        exit 1
    fi

    if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
        log_warn "Dependencies missing. Installing..."
        (cd "${FRONTEND_DIR}" && npm install)
    fi
}

start() {
    cd "${FRONTEND_DIR}"
    log_info "Starting Dev Server..."
    
    # Using --clearScreen false to play nice with other logs
    npm run dev -- --clearScreen false
}

# Main
check_env
start
