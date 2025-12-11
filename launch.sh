#!/bin/bash

# FAANG-style Launch Script for DCA Quant
# -------------------------------------
# This script manages the startup of the DCA Quant full-stack application.
# It handles environment verification, dependency checks, and parallel execution
# of the FastAPI backend and React frontend.

set -e

# ==============================================================================
# Configuration & Constants
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    echo ""
    log_info "Shutting down services..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$SCHEDULER_PID" ]; then
        kill $SCHEDULER_PID 2>/dev/null || true
    fi
    
    # Reset trap to default to avoid infinite loop if called again
    trap - SIGINT SIGTERM EXIT
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGINT SIGTERM EXIT

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

check_backend() {
    log_info "Checking Backend configuration..."
    
    if [ ! -d "$BACKEND_DIR" ]; then
        log_error "Backend directory not found at $BACKEND_DIR"
        exit 1
    fi

    # Check for virtual environment
    if [ -f "$BACKEND_DIR/venv/bin/activate" ]; then
        VENV_ACTIVATE="$BACKEND_DIR/venv/bin/activate"
        log_info "Found venv at: $VENV_ACTIVATE"
    elif [ -f "$BACKEND_DIR/.venv/bin/activate" ]; then
        VENV_ACTIVATE="$BACKEND_DIR/.venv/bin/activate"
        log_info "Found .venv at: $VENV_ACTIVATE"
    else
        log_warn "No virtual environment found in backend/venv or backend/.venv"
        log_warn "Assuming system python or active conda environment."
    fi
}

check_frontend() {
    log_info "Checking Frontend configuration..."
    
    if [ ! -d "$FRONTEND_DIR" ]; then
        log_error "Frontend directory not found at $FRONTEND_DIR"
        exit 1
    fi
    
    if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
        log_warn "node_modules not found in frontend. Dependencies might be missing."
        read -p "Do you want to run 'npm install'? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$FRONTEND_DIR"
            log_info "Installing frontend dependencies..."
            npm install
            cd "$PROJECT_ROOT"
        fi
    fi
}

# ==============================================================================
# Service Launchers
# ==============================================================================

start_backend() {
    log_info "Starting Backend Service..."
    cd "$BACKEND_DIR"
    
    if [ ! -z "$VENV_ACTIVATE" ]; then
        source "$VENV_ACTIVATE"
    fi
    
    # Suppress Pandera import warning
    export DISABLE_PANDERA_IMPORT_WARNING=True
    
    # Install dependencies if requirements.txt has changed (simple check)
    if [ -f "requirements.txt" ]; then
        log_info "Verifying backend dependencies..."
        pip install -r requirements.txt | grep -v "Requirement already satisfied" || true
    fi
    
    # Parse arguments
    UPDATE_SIGNALS="false"
    for arg in "$@"; do
        case $arg in
            signal=true|signal=True|SIGNAL=TRUE)
                UPDATE_SIGNALS="true"
                ;;
            --update-signals) # Legacy support
                UPDATE_SIGNALS="true"
                ;;
        esac
    done

    log_info "Signal Generation: $UPDATE_SIGNALS (Use signal=true to enable)"
    export ENABLE_SIGNALS="$UPDATE_SIGNALS"
    export PYTHONUNBUFFERED=1

    # Run migrations once before starting server
    log_info "Running database migrations..."
    # We use a python one-liner to run the migration function we extracted
    python -c "from app.core.migrations import run_migrations; run_migrations()" || log_warn "Migrations failed, but continuing..."
    
    # Force uvicorn to ensure reload-exclude flags work correctly
    uvicorn main:app --reload --reload-exclude "data/**" --reload-exclude "data_lake/**" --reload-exclude "logs/**" --reload-exclude "mlruns/**" --reload-exclude "*.sqlite*" --reload-exclude "*.parquet" --port 8000 &
    
    BACKEND_PID=$!
    log_success "Backend started (PID: $BACKEND_PID) at http://localhost:8000"
    
    # Start Scheduler/Worker in background
    log_info "Starting Scheduler/Worker..."
    python scheduler.py &
    SCHEDULER_PID=$!
    log_success "Scheduler started (PID: $SCHEDULER_PID)"
    
    cd "$PROJECT_ROOT"
}

start_frontend() {
    log_info "Starting Frontend Service..."
    cd "$FRONTEND_DIR"
    
    # Use --clearScreen false to preserve backend logs
    npm run dev -- --clearScreen false &
    
    FRONTEND_PID=$!
    log_success "Frontend started (PID: $FRONTEND_PID) at http://localhost:5173"
    cd "$PROJECT_ROOT"
}

# ==============================================================================
# Main Execution
# ==============================================================================

main() {
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}   DCA Quant - Launch Control Center    ${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "${BLUE}Startup Pipeline:${NC}"
    echo -e "  1. Seed securities (S&P 500 + Nasdaq 100)"
    echo -e "  2. Data lake catch-up (Parquet)"
    echo -e "  3. Generate signals if stale"
    echo ""
    
    check_backend
    check_frontend
    
    start_backend "$@"
    
    # Wait for backend to complete initialization pipeline
    log_info "Waiting for backend initialization (this may take a moment on first run)..."
    sleep 5
    
    start_frontend
    
    log_success "All services are running. Press Ctrl+C to stop."
    echo ""
    echo -e "${GREEN}Dashboard URLs:${NC}"
    echo -e "  Backend API:  http://localhost:8000"
    echo -e "  Frontend:     http://localhost:5173"
    echo -e "  API Docs:     http://localhost:8000/docs"
    echo ""
    
    # Wait for processes
    wait
}

main "$@"
