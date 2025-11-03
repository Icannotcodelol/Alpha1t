#!/bin/bash
set -euo pipefail

# Run the PPO-LSTM trainer. This expects the package to be installed in a venv.
# Example:
#   ./scripts/train.sh --iterations 1000 --games-per-iter 64 --save-dir runs/ppo

python -m alpha_1000.rl.ppo_lstm.trainer "$@"
