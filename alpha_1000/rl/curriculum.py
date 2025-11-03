"""Curriculum learning for progressive difficulty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .ppo_lstm.selfplay import SelfPlayConfig

__all__ = ["CurriculumStage", "CurriculumSchedule", "create_default_curriculum"]


@dataclass
class CurriculumStage:
    """Single stage of curriculum with specific training configuration."""
    
    name: str
    start_iteration: int
    games_per_iter: int
    description: str
    full_game: bool = False
    
    def to_selfplay_config(self) -> SelfPlayConfig:
        """Convert to SelfPlayConfig."""
        return SelfPlayConfig(
            games_per_iteration=self.games_per_iter,
            full_game=self.full_game,
            curriculum_stage=self.start_iteration,
        )


@dataclass
class CurriculumSchedule:
    """Complete curriculum schedule."""
    
    stages: Dict[int, CurriculumStage]
    
    def get_stage(self, iteration: int) -> CurriculumStage:
        """Get appropriate stage for given iteration."""
        # Find the latest stage that has started
        applicable_stages = [
            (start_iter, stage) 
            for start_iter, stage in self.stages.items() 
            if start_iter <= iteration
        ]
        
        if not applicable_stages:
            # Return earliest stage
            earliest = min(self.stages.keys())
            return self.stages[earliest]
        
        # Return stage with highest start iteration <= current iteration
        _, stage = max(applicable_stages, key=lambda x: x[0])
        return stage
    
    def get_config(self, iteration: int) -> SelfPlayConfig:
        """Get SelfPlayConfig for given iteration."""
        stage = self.get_stage(iteration)
        return stage.to_selfplay_config()


def create_default_curriculum() -> CurriculumSchedule:
    """Create default progressive curriculum.
    
    Stage progression:
    1. Warmup (0-500): Learn basic card play, few games
    2. Foundation (500-2000): More games, stabilize learning
    3. Advanced (2000-5000): Full complexity, many games  
    4. Expert (5000+): Full games to 1000 points
    """
    
    stages = {
        0: CurriculumStage(
            name="warmup",
            start_iteration=0,
            games_per_iter=4,
            full_game=False,
            description="Learn basic trick-taking with small batches",
        ),
        500: CurriculumStage(
            name="foundation",
            start_iteration=500,
            games_per_iter=8,
            full_game=False,
            description="Scale up experience collection",
        ),
        2000: CurriculumStage(
            name="advanced",
            start_iteration=2000,
            games_per_iter=16,
            full_game=False,
            description="High-volume single-hand training",
        ),
        5000: CurriculumStage(
            name="expert",
            start_iteration=5000,
            games_per_iter=8,
            full_game=True,
            description="Full game training for strategic depth",
        ),
    }
    
    return CurriculumSchedule(stages=stages)


def create_fast_curriculum() -> CurriculumSchedule:
    """Create faster curriculum for testing/shorter runs."""
    
    stages = {
        0: CurriculumStage(
            name="warmup",
            start_iteration=0,
            games_per_iter=4,
            full_game=False,
            description="Quick warmup",
        ),
        100: CurriculumStage(
            name="scaling",
            start_iteration=100,
            games_per_iter=8,
            full_game=False,
            description="Scale up quickly",
        ),
        500: CurriculumStage(
            name="full_game",
            start_iteration=500,
            games_per_iter=4,
            full_game=True,
            description="Full game training",
        ),
    }
    
    return CurriculumSchedule(stages=stages)

