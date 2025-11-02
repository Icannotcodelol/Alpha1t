#!/bin/bash
set -euo pipefail
pytest tests -v --cov=engine --cov-report=term-missing
