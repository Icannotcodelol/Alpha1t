#!/bin/bash
set -euo pipefail
python -m alpha-1000.rl.ppo_lstm.trainer "$@"
