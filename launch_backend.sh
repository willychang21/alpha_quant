#!/bin/bash
# ==============================================================================
# DCA Quant - Backend Launcher
# ==============================================================================

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log_info() { echo -e "\033[0;90m[$(timestamp)]\033[0m ${BLUE}BACKEND${NC}  $1"; }
log_success() { echo -e "\033[0;90m[$(timestamp)]\033[0m ${GREEN}BACKEND${NC}  $1"; }
log_warn() { echo -e "\033[0;90m[$(timestamp)]\033[0m ${YELLOW}BACKEND${NC}  $1"; }
log_error() { echo -e "\033[0;90m[$(timestamp)]\033[0m ${RED}BACKEND${NC}  $1" >&2; }

# Environment Check
check_env() {
    if [[ ! -d "${BACKEND_DIR}" ]]; then
        log_error "Directory not found: ${BACKEND_DIR}"
        exit 1
    fi

    # Activate Virtual Environment
    if [[ -f "${BACKEND_DIR}/venv/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "${BACKEND_DIR}/venv/bin/activate"
    elif [[ -f "${BACKEND_DIR}/.venv/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "${BACKEND_DIR}/.venv/bin/activate"
    else
        log_warn "No virtual environment found. Using system Python."
    fi
}

start() {
    local enable_signals="${1:-false}"
    
    cd "${BACKEND_DIR}"
    
    # Environment Variables
    export DISABLE_PANDERA_IMPORT_WARNING=True
    export ENABLE_SIGNALS="${enable_signals}"
    export PYTHONUNBUFFERED=1
    
    log_info "Initializing..."
    
    # Database Migrations
    if ! python -c "from app.core.migrations import run_migrations; run_migrations()"; then
        log_warn "Migrations check encountered an issue (non-critical)..."
    fi

    # Start Scheduler (Background)
    python scheduler.py &
    SCHEDULER_PID=$!
    log_success "Scheduler started (PID: ${SCHEDULER_PID})"
    
    # Start Uvicorn (Foreground, or managed by caller? Launcher usually expects calling script to handle bg, 
    # but here we are the runner. If we run uvicorn in output, this script blocks.
    # To keep parity with launch.sh which backgrounds both, this script could block OR background.
    # Since we want separate shells, let's keep this blocking so it can be managed by a supervisor or launch.sh & wait.
    # BUT launch.sh needs to kill scheduler if this script dies. 
    # Better pattern: Trap EXIT in this script to kill scheduler.
    
    trap "kill ${SCHEDULER_PID} 2>/dev/null" EXIT

    log_info "Starting API Server..."
    uvicorn main:app \
        --reload \
        --reload-exclude "data/**" \
        --reload-exclude "data_lake/**" \
        --reload-exclude "logs/**" \
        --reload-exclude "mlruns/**" \
        --reload-exclude "*.sqlite*" \
        --reload-exclude "*.parquet" \
        --port 8000
}

# Main
check_env
# Parse args: accept --signal or signal=true
ENABLE_SIGNALS="false"
for arg in "$@"; do
    case $arg in
        --signal|signal=true|signal=True) ENABLE_SIGNALS="true" ;;
    esac
done

start "${ENABLE_SIGNALS}"
