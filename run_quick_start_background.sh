#!/usr/bin/env bash
# Run quick_start_compliance.py in the background. Logs to quick_start_compliance.log.
# Usage: ./run_quick_start_background.sh   (or: bash run_quick_start_background.sh)

cd "$(dirname "$0")"
LOG=quick_start_compliance.log
echo "Starting quick_start_compliance.py in background. Log: $LOG"
nohup python3 quick_start_compliance.py > "$LOG" 2>&1 &
echo "PID: $!. To watch: tail -f $LOG"
