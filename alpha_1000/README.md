# Alpha-1000: Professional Tysiąc (Thousand) Card Game Engine

Alpha-1000 is an end-to-end implementation of the Polish card game Tysiąc featuring a
configurable engine, a polished Streamlit UI, multiple bot opponents, and a PPO-LSTM
reinforcement learning agent. The project is designed for extensibility and research
experimentation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
streamlit run ui/app.py
```

## Repository Layout

See `docs/architecture.md` for a detailed overview of the modules, data flow, and key
interfaces.

## License

MIT
