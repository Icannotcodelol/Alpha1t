"""Dense reward shaping for better learning signals."""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.cards import Card, CARD_POINTS
from ..engine.state import GameState, PlayerID

__all__ = ["DenseRewardConfig", "compute_dense_reward"]


@dataclass
class DenseRewardConfig:
    """Configuration for dense reward shaping."""
    
    # Card play rewards
    trick_win_bonus: float = 0.1          # Win a trick
    trick_loss_penalty: float = -0.05     # Lose a trick
    high_value_card_bonus: float = 0.02   # Play 10 or Ace
    point_value_scale: float = 0.01       # Scale card points
    
    # Bidding rewards
    reasonable_bid_bonus: float = 0.05    # Bid within hand strength
    overbid_penalty: float = -0.1         # Bid too high
    pass_when_weak_bonus: float = 0.02    # Pass with weak hand
    
    # Bombing rewards
    good_bomb_bonus: float = 0.2          # Bomb unwinnable contract
    bad_bomb_penalty: float = -0.3        # Waste bomb on winnable hand
    
    # Contract rewards
    contract_success: float = 1.0         # Achieve contract
    contract_fail: float = -1.0           # Fail contract
    contract_margin_bonus: float = 0.005  # Extra points beyond contract
    
    # Meld rewards
    meld_declared_bonus: float = 0.1      # Declare a meld
    meld_value_scale: float = 0.001       # Scale meld points
    
    # Game-level rewards (for full game training)
    game_win: float = 5.0                 # Win the game
    game_loss: float = -5.0               # Lose the game
    progress_bonus: float = 0.001         # Scale score progress


def compute_dense_reward(
    prev_state: GameState | None,
    action_type: str,
    action_value: int | None,
    curr_state: GameState,
    player: PlayerID,
    config: DenseRewardConfig | None = None,
) -> float:
    """Compute dense reward for a single action.
    
    Args:
        prev_state: State before action (None if first action)
        action_type: Type of action ('bid', 'bomb', 'play', 'meld')
        action_value: Value of action (bid amount, card index, etc.)
        curr_state: State after action
        player: Player who took action
        config: Reward configuration
        
    Returns:
        Dense reward value
    """
    cfg = config or DenseRewardConfig()
    reward = 0.0
    
    if action_type == "play":
        reward += _card_play_reward(prev_state, curr_state, player, cfg)
    elif action_type == "bid":
        reward += _bidding_reward(prev_state, curr_state, player, action_value, cfg)
    elif action_type == "bomb":
        reward += _bombing_reward(curr_state, player, action_value, cfg)
    elif action_type == "meld":
        reward += _meld_reward(curr_state, player, cfg)
    
    # Check for hand completion
    if _hand_completed(prev_state, curr_state):
        reward += _hand_completion_reward(curr_state, player, cfg)
    
    # Check for game completion
    if _game_completed(curr_state):
        reward += _game_completion_reward(curr_state, player, cfg)
    
    return reward


def _card_play_reward(
    prev_state: GameState | None,
    curr_state: GameState,
    player: PlayerID,
    cfg: DenseRewardConfig,
) -> float:
    """Reward for card playing decisions."""
    reward = 0.0
    
    if not curr_state.trick_history:
        return reward
    
    current_trick = curr_state.trick_history[-1]
    
    # Check if trick is complete
    if len(current_trick) == 2:
        winner = curr_state.playing_player or 0
        
        # Calculate trick value
        trick_value = sum(CARD_POINTS[card.rank] for _, card in current_trick)
        
        if winner == player:
            # Won the trick
            reward += cfg.trick_win_bonus
            reward += trick_value * cfg.point_value_scale
        else:
            # Lost the trick
            reward += cfg.trick_loss_penalty
    
    # Bonus for playing high-value cards strategically
    if prev_state and len(prev_state.hands[player]) > len(curr_state.hands[player]):
        # Player just played a card
        for pid, card in current_trick:
            if pid == player:
                if card.points >= 10:  # 10 or Ace
                    reward += cfg.high_value_card_bonus
                break
    
    return reward


def _bidding_reward(
    prev_state: GameState | None,
    curr_state: GameState,
    player: PlayerID,
    bid_value: int | None,
    cfg: DenseRewardConfig,
) -> float:
    """Reward for bidding decisions."""
    reward = 0.0
    
    if bid_value is None:
        # Passed - check if it was a good pass
        if prev_state and player in prev_state.hands:
            hand_strength = _estimate_hand_strength(prev_state.hands[player])
            if hand_strength < 100:  # Weak hand
                reward += cfg.pass_when_weak_bonus
    else:
        # Made a bid - check if reasonable
        if prev_state and player in prev_state.hands:
            hand_strength = _estimate_hand_strength(prev_state.hands[player])
            diff = abs(bid_value - hand_strength)
            
            if diff < 20:  # Close to hand strength
                reward += cfg.reasonable_bid_bonus
            elif bid_value > hand_strength + 30:  # Overbid
                reward += cfg.overbid_penalty
    
    return reward


def _bombing_reward(
    curr_state: GameState,
    player: PlayerID,
    bombed: int | None,
    cfg: DenseRewardConfig,
) -> float:
    """Reward for bombing decisions."""
    reward = 0.0
    
    if bombed == 0:  # Chose to bomb
        # Estimate if bomb was good (contract likely unwinnable)
        if curr_state.current_bid:
            hand_strength = _estimate_hand_strength(curr_state.hands.get(player, []))
            if curr_state.current_bid > hand_strength + 40:
                # Good bomb - contract was likely impossible
                reward += cfg.good_bomb_bonus
            else:
                # Bad bomb - might have been winnable
                reward += cfg.bad_bomb_penalty
    
    return reward


def _meld_reward(
    curr_state: GameState,
    player: PlayerID,
    cfg: DenseRewardConfig,
) -> float:
    """Reward for meld declarations."""
    reward = 0.0
    
    # Check if player declared a new meld
    if curr_state.meld_history:
        last_meld_player, king, queen = curr_state.meld_history[-1]
        if last_meld_player == player:
            reward += cfg.meld_declared_bonus
            
            # Bonus based on meld value
            from ..engine.marriages import MARRIAGE_POINTS
            meld_value = MARRIAGE_POINTS.get(king.suit, 0)
            reward += meld_value * cfg.meld_value_scale
    
    return reward


def _hand_completion_reward(
    curr_state: GameState,
    player: PlayerID,
    cfg: DenseRewardConfig,
) -> float:
    """Reward for hand completion (all tricks played)."""
    reward = 0.0
    
    # Count points scored by player
    player_points = 0
    for trick in curr_state.trick_history:
        if len(trick) == 2:
            winner = _determine_trick_winner(trick, curr_state)
            if winner == player:
                player_points += sum(CARD_POINTS[card.rank] for _, card in trick)
    
    # Add meld points
    for meld_player, king, queen in curr_state.meld_history:
        if meld_player == player:
            from ..engine.marriages import MARRIAGE_POINTS
            player_points += MARRIAGE_POINTS.get(king.suit, 0)
    
    # Check if player was declarer
    if curr_state.playing_player == player and curr_state.current_bid:
        if player_points >= curr_state.current_bid:
            # Made contract
            reward += cfg.contract_success
            margin = player_points - curr_state.current_bid
            reward += margin * cfg.contract_margin_bonus
        else:
            # Failed contract
            reward += cfg.contract_fail
    else:
        # Defender - just get points scaled
        reward += player_points * cfg.point_value_scale * 0.5
    
    return reward


def _game_completion_reward(
    curr_state: GameState,
    player: PlayerID,
    cfg: DenseRewardConfig,
) -> float:
    """Reward for game completion (someone reached 1000)."""
    reward = 0.0
    
    player_score = curr_state.scores.get(player, 0)
    opponent_score = curr_state.scores.get(1 - player, 0)
    
    if player_score >= 1000:
        reward += cfg.game_win
    elif opponent_score >= 1000:
        reward += cfg.game_loss
    
    # Progress bonus
    reward += player_score * cfg.progress_bonus
    
    return reward


def _hand_completed(prev_state: GameState | None, curr_state: GameState) -> bool:
    """Check if a hand just completed."""
    if prev_state is None:
        return False
    
    # Hand completes when all cards are played
    prev_cards = sum(len(hand) for hand in prev_state.hands.values())
    curr_cards = sum(len(hand) for hand in curr_state.hands.values())
    
    return prev_cards > 0 and curr_cards == 0


def _game_completed(curr_state: GameState) -> bool:
    """Check if game is complete."""
    return any(score >= curr_state.rules.game.target_score 
               for score in curr_state.scores.values())


def _estimate_hand_strength(cards: list[Card]) -> int:
    """Estimate hand strength for bidding/bombing decisions."""
    if not cards:
        return 0
    
    # Simple estimate based on card points
    total_points = sum(card.points for card in cards)
    
    # Count high cards
    high_cards = sum(1 for card in cards if card.points >= 10)
    
    # Check for melds
    from ..engine.marriages import find_marriages, MARRIAGE_POINTS
    marriages = find_marriages(cards)
    meld_value = sum(MARRIAGE_POINTS[suit] for suit, _, _ in marriages)
    
    # Estimate: base points + high card bonus + meld value
    estimate = total_points * 2 + high_cards * 10 + meld_value
    
    # Clip to reasonable range
    return max(100, min(300, estimate))


def _determine_trick_winner(trick: list[tuple[PlayerID, Card]], state: GameState) -> PlayerID:
    """Determine winner of a trick (simplified)."""
    if len(trick) != 2:
        return trick[0][0] if trick else 0
    
    from ..engine.cards import RANK_ORDER
    
    lead_player, lead_card = trick[0]
    follow_player, follow_card = trick[1]
    
    # Simple comparison - just check if follower beat leader
    if follow_card.suit == lead_card.suit:
        if RANK_ORDER.index(follow_card.rank) > RANK_ORDER.index(lead_card.rank):
            return follow_player
    
    return lead_player

