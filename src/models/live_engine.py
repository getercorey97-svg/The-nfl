import logging
from typing import Dict, Optional
import numpy as np
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveEngine")

class LivePredictiveEngine:
    """Computes real-time win probabilities, in-game spreads, live totals, and live prop projections."""

    # League-average Expected Points (EP) lookup by down and field position (yards to opponent end zone)
    @staticmethod
    def calculate_expected_points(down: int, distance: int, yardline_from_goal: int) -> float:
        """Estimates the expected points of the current possession situation."""
        # Baseline linear-exponential field position curve
        base_ep = (100 - yardline_from_goal) * 0.06 - 1.5
        down_penalty = (down - 1) * 0.75 + (distance * 0.05)
        return float(np.clip(base_ep - down_penalty, -2.0, 6.0))

    @classmethod
    def calculate_live_win_probability(
        cls,
        margin: int,
        seconds_remaining: int,
        pregame_spread: float = 0.0,
        possession_team: Optional[str] = None,
        home_team: Optional[str] = None,
        down: int = 1,
        distance: int = 10,
        yardline_from_goal: int = 50
    ) -> Dict[str, float]:
        """Calculates live home win probability and fair in-game margin using continuous Gaussian decay."""
        t_fraction = max(seconds_remaining, 1) / 3600.0

        # Continuous drift: as t -> 0, pregame spread impact decays to 0
        expected_drift = pregame_spread * t_fraction

        # Situation impact (possession equity)
        ep = cls.calculate_expected_points(down, distance, yardline_from_goal)
        possession_adjustment = ep if possession_team == home_team else -ep

        # Expected final margin
        fair_live_margin = margin + expected_drift + (possession_adjustment * 0.3)

        # NFL single-game variance decays proportionally to sqrt(time remaining)
        # Standard deviation for a full 60-min NFL game is ~13.5 points
        sigma_t = max(13.5 * np.sqrt(t_fraction), 0.5)

        # Home Win Probability = P(Final Margin > 0)
        z = fair_live_margin / sigma_t
        home_wp = float(norm.cdf(z))

        return {
            "home_win_prob": round(home_wp, 4),
            "away_win_prob": round(1.0 - home_wp, 4),
            "fair_live_margin": round(fair_live_margin, 1),
            "live_spread_projection": round(-fair_live_margin, 1)
        }

    @staticmethod
    def calculate_live_total(
        current_total: int,
        seconds_remaining: int,
        pregame_total: float = 45.5,
        down: int = 1,
        distance: int = 10,
        yardline_from_goal: int = 50
    ) -> float:
        """Projects the live end-of-game total based on elapsed pace and remaining game duration."""
        t_fraction = max(seconds_remaining, 0) / 3600.0
        
        # Expected points scored in remaining time based on baseline scoring rate
        remaining_expected_points = pregame_total * t_fraction

        # Include situational Expected Points for the active drive
        current_drive_ep = max(0.0, (100 - yardline_from_goal) * 0.05)
        
        projected_final_total = current_total + remaining_expected_points + current_drive_ep
        return round(float(projected_final_total), 1)

    @staticmethod
    def calculate_live_player_prop(
        current_stat: float,
        pregame_projected: float,
        seconds_remaining: int,
        live_line: float,
        team_is_leading: bool,
        prop_type: str = "passing_yards"
    ) -> Dict[str, float]:
        """Projects player prop final production with game-script volume shifts."""
        t_fraction = max(seconds_remaining, 0) / 3600.0

        # Game script multiplier: Trailing teams pass ~18% more; leading teams run ~22% more
        script_multiplier = 1.0
        if "pass" in prop_type or "rec" in prop_type:
            script_multiplier = 0.85 if team_is_leading else 1.18
        elif "rush" in prop_type:
            script_multiplier = 1.22 if team_is_leading else 0.80

        expected_remaining = pregame_projected * t_fraction * script_multiplier
        live_projected_final = current_stat + expected_remaining

        # Volatility scales with remaining duration
        sigma = max(12.0 * np.sqrt(t_fraction), 2.0)
        z = (live_line - live_projected_final) / sigma
        under_prob = float(norm.cdf(z))
        over_prob = 1.0 - under_prob

        return {
            "current_stat": current_stat,
            "projected_final": round(live_projected_final, 1),
            "live_line": live_line,
            "over_prob": round(over_prob, 4),
            "under_prob": round(under_prob, 4),
            "live_edge": round(live_projected_final - live_line, 1)
        }
