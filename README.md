# MASTER PROMPT — Alpha-1000 (Tysiąc 1v1) — Engine + UI + PPO-LSTM Agent

You are a senior Python engineer. Build a **production-ready** project called **alpha-1000** that implements the **2-player** Polish card game **Tysiąc** (Thousand) exactly as specified below, with a polished Streamlit UI, comprehensive tests, baseline bots, and a PPO-LSTM agent for self-play training. Focus on code quality, maintainability, and accurate game mechanics.

---

## 1) AUTHORITATIVE RULES (implement exactly)

### Core Game Elements

**Deck:** 24 cards (ranks per suit: 9, J, Q, K, 10, A)
**Card points:** 9=0, J=2, Q=3, K=4, 10=10, A=11
**Ranks high→low in a suit:** A > 10 > K > Q > J > 9

### Deal (2 players)
* 10 cards each, 4 cards remain as **two face-down musiki** (2+2)
* Dealer alternates each hand
* Track dealer position in game state

### Bidding (Auction)
* Starts at **100**, increments of **10**, or **Pass**
* Once passed, cannot re-enter bidding
* Highest bid wins; winner becomes the **playing player**
* **Proof requirement:** any **bid > 120** may be challenged:
  - Bidder must **show a meld** (K+Q of same suit) with **value ≥ (bid − 100)**
  - Meld values: ♠=40, ♣=60, ♦=80, ♥=100
  - Showing proof during bidding **does not set trump** and **does not consume** the meld
  - If cannot show valid proof when challenged → automatic loss of the hand

### Musik Phase
* Auction winner chooses **one musik** (2 cards), reveals it
* Takes both cards (hand→12 cards)
* **Bombing opportunity** (see below)
* If not bombing: returns **exactly 2 cards** face-down to form discard pile
* Both players have 10 cards for play phase

### Bombing (Bomba) Rule
* **When:** After viewing chosen musik, before returning any cards
* **Who:** Only the playing player (auction winner) may bomb
* **Effect:** Abandons the hand completely:
  - All cards collected and reshuffled
  - New deal with same dealer
  - No points scored or lost
  - No contract penalties
* **Limit:** Each player has **2 bombs per game** (track usage)
* **Strategy:** Escape mechanism for unwinnable contracts after bad musik

### Trump & Melds (Dynamic)
* Initially **no trump**
* **Declaring a meld:** Only by **trick leader** when **playing one of the pair**
* Shows the other card, immediately **sets trump to that suit**
* **Any later meld immediately overrides trump** (by either player)
* Each player may declare **each suit at most once** per hand
* Meld points scored by declarer

### Trick Play
* Playing player leads first trick
* **Must follow suit** if possible
* **Must overtake** if following suit and able to beat current winner (optional: make configurable)
* **Must overtrump** if trump winning and you can't follow but have higher trump (optional: make configurable)
* If cannot follow suit and not required to trump: play any card
* Trick won by: highest trump if any, else highest card of led suit
* Meld declaration changes trump **before** current trick resolution

### Scoring Per Hand
* Each player tallies **card points + meld points** from won tricks
* **Defender:** adds total **rounded up to nearest 10** to game score
* **Playing player:** declares a **contract** (must be ≥ winning bid, multiple of 10)
  - **Achieved:** add contract to game score
  - **Failed:** subtract contract from game score
* **800-lock:** At ≥800 points, can only advance as playing player
* **Game end:** First to ≥1000 wins

### Configurable Rules
All rules configurable via `rules_tysiac.yaml`:
```yaml
game:
  target_score: 1000
  lock_score: 800
  enable_lock: true
  
bidding:
  start_bid: 100
  increment: 10
  proof_threshold: 120
  
bombing:
  enabled: true
  bombs_per_player: 2
  
trick_rules:
  must_overtake: true  # Traditional varies by region
  must_overtrump: true  # Traditional varies by region
```

---

## 2) PROJECT STRUCTURE

```
alpha-1000/
├── engine/
│   ├── __init__.py
│   ├── cards.py                # Suit/Rank/Card classes, ordering, points
│   ├── rules.py                # RulesConfig class with Pydantic validation
│   ├── rules_tysiac.yaml       # Default rule configuration
│   ├── state.py                # GameState with full history tracking
│   ├── phases/
│   │   ├── __init__.py
│   │   ├── bidding.py          # Auction logic, proof validation
│   │   ├── musik.py            # Musik selection, bombing decision
│   │   ├── play.py             # Trick play logic
│   │   └── scoring.py          # Point calculation, contract logic
│   ├── actions.py              # Action validation and masking
│   ├── marriages.py            # Meld declaration and trump management
│   └── game.py                 # Main game loop orchestrator
│
├── bots/
│   ├── __init__.py
│   ├── base_bot.py             # Abstract bot interface
│   ├── random_bot.py           # Baseline random legal moves
│   ├── greedy_bot.py           # Simple greedy strategy
│   ├── heuristic_bot.py        # Advanced heuristics
│   └── arena.py                # Bot tournament system
│
├── ui/
│   ├── __init__.py
│   ├── app.py                  # Main Streamlit app
│   ├── components/
│   │   ├── hand_display.py     # Card hand visualization
│   │   ├── bidding_panel.py    # Auction interface with proof UI
│   │   ├── trick_viewer.py     # Current and past tricks
│   │   ├── game_status.py      # Scores, trump, bombs remaining
│   │   └── game_log.py         # Move history and melds
│   └── assets/
│       └── cards/               # Card images (SVG preferred)
│
├── rl/
│   ├── __init__.py
│   ├── ppo_lstm/
│   │   ├── __init__.py
│   │   ├── network.py          # Neural network architecture
│   │   ├── agent.py            # PPO-LSTM agent implementation
│   │   ├── buffer.py           # Experience replay buffer
│   │   ├── trainer.py          # Training loop and optimization
│   │   ├── selfplay.py         # Parallel self-play workers
│   │   └── evaluator.py        # Performance evaluation
│   ├── encoding.py             # State/action encoding for NN
│   └── rewards.py              # Reward shaping strategies
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_cards.py
│   │   ├── test_bidding.py
│   │   ├── test_bombing.py     # Comprehensive bombing tests
│   │   ├── test_melds.py
│   │   ├── test_scoring.py
│   │   └── test_tricks.py
│   ├── integration/
│   │   ├── test_full_game.py
│   │   ├── test_edge_cases.py
│   │   └── test_determinism.py
│   └── fixtures/
│       ├── hands.py            # Predefined test hands
│       └── scenarios.json      # Complex game scenarios
│
├── data/
│   ├── checkpoints/            # Model checkpoints
│   ├── logs/                   # Training logs
│   └── replays/                # Game replays for analysis
│
├── scripts/
│   ├── setup.sh                # One-click setup
│   ├── test.sh                 # Run all tests
│   ├── train.sh                # Start training
│   └── benchmark.py            # Performance benchmarking
│
├── docs/
│   ├── rules.md                # Complete game rules
│   ├── architecture.md         # System design documentation
│   └── api.md                  # Module API reference
│
├── pyproject.toml              # Modern Python packaging
├── requirements.txt            # Dependencies
├── README.md                   # Project overview and quickstart
├── CHANGELOG.md                # Version history
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions CI/CD

```

---

## 3) ENHANCED IMPLEMENTATION REQUIREMENTS

### Code Quality Standards
* Python 3.11+ with full type hints (use `typing` and `typing_extensions`)
* Docstrings for all public functions (Google style)
* Maximum function length: 30 lines (refactor if longer)
* Maximum file length: 300 lines
* Use `dataclasses` or Pydantic for all data structures
* Implement `__repr__` for debugging on all classes

### Error Handling
* Custom exceptions hierarchy:
  ```python
  class TysiacError(Exception): pass
  class InvalidActionError(TysiacError): pass
  class InvalidBidError(InvalidActionError): pass
  class BombingError(TysiacError): pass
  ```
* Validate all inputs at boundaries
* Log errors with context (use `logging` module)

### Performance Optimizations
* Use `numpy` arrays for card representations
* Implement move caching for legal move generation
* Batch neural network inference
* Profile bottlenecks with `cProfile`

### Testing Requirements
* Minimum 90% code coverage
* Property-based tests with `hypothesis` for game logic
* Deterministic tests with fixed seeds
* Performance benchmarks (< 1ms per move validation)

---

## 4) COMPREHENSIVE TEST SUITE

### Critical Test Cases

1. **Bombing Mechanics**
   ```python
   def test_bombing_after_musik_view():
       # Player wins bid, views musik, decides to bomb
       # Verify: hand abandoned, cards reshuffled, same dealer
   
   def test_bombing_limit_enforcement():
       # Player uses 2 bombs, tries third
       # Verify: third bomb rejected
   
   def test_no_bombing_after_return():
       # Player returns cards, then tries to bomb
       # Verify: bombing no longer allowed
   ```

2. **Bidding Edge Cases**
   ```python
   def test_proof_exactly_at_threshold():
       # bid=130, shows ♣K+Q (60) → OK (30 ≤ 60)
   
   def test_challenge_timing():
       # Verify challenge only possible before musik phase
   ```

3. **Trump Dynamics**
   ```python
   def test_meld_override_same_trick():
       # P1 declares ♦, P2 plays, then declares ♥
       # Verify: ♥ is trump for trick resolution
   ```

4. **800-Lock Scenarios**
   ```python
   def test_defender_at_800_stays():
       # Defender at 800, gets 62→70 points
       # Verify: stays at 800, not 870
   ```

---

## 5) STREAMLIT UI ENHANCEMENTS

### Main Features
* **Game Modes:** Human vs Bot, Bot vs Bot, Online Multiplayer (stretch)
* **Difficulty Levels:** Random, Greedy, Heuristic, Trained RL
* **Visual Polish:**
  - Smooth card animations (CSS transitions)
  - Drag-and-drop for card play
  - Sound effects (optional toggle)
  - Dark/light theme
* **Statistics Dashboard:**
  - Win rates per position
  - Average contract values
  - Bombing frequency analysis
* **Replay Viewer:** Load and step through saved games
* **Rule Customization:** In-app rule editor with presets

### UI Components Structure
```python
# ui/app.py
def main():
    st.set_page_config(page_title="Alpha-1000", layout="wide")
    
    # Sidebar: game settings, bot selection, rules
    with st.sidebar:
        game_mode = st.selectbox("Mode", ["Human vs Bot", "Bot vs Bot"])
        bot_difficulty = st.selectbox("Bot", ["Random", "Greedy", "PPO-LSTM"])
        
    # Main area: game board
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        show_opponent_info()
    with col2:
        show_game_board()  # Trick area, trump indicator
    with col3:
        show_player_info()
    
    # Bottom: player's hand
    show_player_hand()
```

---

## 6) PPO-LSTM AGENT IMPROVEMENTS

### Architecture Enhancements
```python
class TysiacNetwork(nn.Module):
    def __init__(self, hidden_size=256, lstm_layers=2):
        super().__init__()
        # Separate encoders for different game aspects
        self.hand_encoder = nn.Linear(24, 64)
        self.trick_encoder = nn.Linear(48, 64)
        self.history_encoder = nn.LSTM(128, hidden_size, lstm_layers)
        
        # Multi-head attention for card relationships
        self.attention = nn.MultiheadAttention(hidden_size, 4)
        
        # Separate heads with different architectures
        self.bid_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 20)  # Possible bids
        )
        self.play_head = nn.Linear(hidden_size, 24)  # Card actions
        self.bomb_head = nn.Linear(hidden_size, 2)   # Bomb decision
        self.value_head = nn.Linear(hidden_size, 1)
```

### Training Improvements
* **Curriculum Learning:** Start with simplified rules, gradually add complexity
* **Self-play League:** Maintain population of past versions
* **Reward Shaping:**
  - Small rewards for valid melds (+0.01)
  - Penalty for wasted bombs (-0.05)
  - Bonus for precise contracts (+0.02)
* **Hyperparameter Schedule:** Decay learning rate and entropy coefficient

### Evaluation Metrics
* Win rate vs each bot tier
* Average game score
* Contract success rate
* Bombing efficiency (successful escapes vs wasted bombs)
* Meld timing optimality

---

## 7) DEVELOPMENT WORKFLOW

### Initial Development Order
1. **Core Engine** (2-3 days)
   - Cards and basic structures
   - State management
   - Rule validation
   
2. **Game Phases** (2 days)
   - Bidding with proof
   - Musik and bombing
   - Trick play
   
3. **Basic UI** (1 day)
   - Minimal playable interface
   - Human vs random bot
   
4. **Testing Suite** (1 day)
   - Unit tests for all components
   - Integration tests for full games
   
5. **Bot Development** (1 day)
   - Baseline bots
   - Heuristic strategies
   
6. **RL Agent** (2-3 days)
   - Network implementation
   - Training pipeline
   - Evaluation framework

### Git Workflow
```bash
# Feature branches
git checkout -b feature/bombing-rule
# Make changes
git add -A && git commit -m "feat: implement bombing mechanism"
# Test thoroughly
pytest tests/ --cov=engine
# Merge when green
git checkout main && git merge feature/bombing-rule
```

---

## 8) COMMANDS AND SCRIPTS

### Setup and Run
```bash
# One-click setup (scripts/setup.sh)
#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest tests/unit -v
echo "✅ Setup complete! Run 'streamlit run ui/app.py' to start."

# Development commands
pytest tests/ -v --cov=engine --cov-report=html  # Full test suite
python -m cProfile -s cumtime engine/game.py     # Performance profiling
python scripts/benchmark.py --games 1000         # Benchmark engine

# Training
python rl/ppo_lstm/trainer.py \
    --workers 8 \
    --batch-size 512 \
    --learning-rate 3e-4 \
    --entropy-coef 0.01 \
    --checkpoint-interval 10000

# Evaluation
python rl/ppo_lstm/evaluator.py \
    --checkpoint data/checkpoints/best.pt \
    --opponents all \
    --games 1000
```

### Docker Support
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0"]
```

---

## 9) README TEMPLATE

```markdown
# Alpha-1000: Professional Tysiąc (Thousand) Card Game Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)]()

## Features
- ✅ Complete 2-player Tysiąc implementation with configurable rules
- ✅ Bombing (Bomba) mechanism - escape bad hands
- ✅ Beautiful Streamlit UI with animations
- ✅ PPO-LSTM self-learning agent
- ✅ Multiple bot difficulty levels
- ✅ Comprehensive test coverage

## Quick Start
\`\`\`bash
git clone https://github.com/username/Alpha1t.git
cd alpha-1000
./scripts/setup.sh
streamlit run ui/app.py
\`\`\`

## Game Rules
[Include complete rules with bombing explanation]

## Architecture
[Brief overview with diagram]

## Training the AI
\`\`\`bash
python rl/ppo_lstm/trainer.py --workers 8
\`\`\`

## Contributing
See CONTRIBUTING.md for guidelines.

## License
MIT
```

---

## 10) CRITICAL IMPLEMENTATION NOTES

### Bombing Implementation Details
```python
@dataclass
class BombingState:
    """Tracks bombing availability and history"""
    bombs_remaining: Dict[PlayerID, int]  # {player_id: bombs_left}
    bomb_history: List[BombEvent]         # When each bomb was used
    current_bomb_window: bool              # True only after musik view
    
class MusikPhase:
    def execute(self, state: GameState) -> GameState:
        # 1. Player views chosen musik
        state = self.reveal_musik(state)
        
        # 2. BOMBING DECISION POINT
        if state.rules.bombing_enabled:
            state.bombing_state.current_bomb_window = True
            decision = self.get_bomb_decision(state)
            
            if decision.bomb:
                if state.bombing_state.bombs_remaining[state.playing_player] <= 0:
                    raise BombingError("No bombs remaining")
                
                # Execute bomb: abandon hand, reshuffle, keep same dealer
                return self.execute_bomb(state)
            
            state.bombing_state.current_bomb_window = False
        
        # 3. Return cards (no bombing after this point)
        state = self.return_cards(state)
        return state
```

### Performance Considerations
* Cache legal moves per state
* Use numpy for card operations
* Implement state hashing for transposition tables
* Profile with large-scale bot tournaments

### Testing Philosophy
* Every rule edge case needs a test
* Use hypothesis for property testing game invariants
* Regression tests for any reported bugs
* Benchmark tests to prevent performance degradation

---

**END OF MASTER PROMPT**

This improved version:
1. **Fixes the bombing rule** to match traditional Tysiąc
2. **Adds comprehensive testing** for the bombing mechanism
3. **Improves project structure** with better organization
4. **Enhances the RL agent** with modern techniques
5. **Adds performance optimizations** and profiling
6. **Includes UI polish** and user experience improvements
7. **Provides clear development workflow** and commands
8. **Adds Docker support** for easy deployment
9. **Includes proper error handling** and logging
10. **Emphasizes code quality** with style guidelines

The key improvement is treating bombing as a strategic escape mechanism rather than a doubling mechanism, which fundamentally changes the game's strategy and makes it true to the original Tysiąc rules.
