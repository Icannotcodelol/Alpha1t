#!/bin/bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest tests/unit -v || true
echo "✅ Setup complete! Run 'streamlit run ui/app.py' to start."
