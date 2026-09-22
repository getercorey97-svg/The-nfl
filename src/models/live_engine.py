import logging
from typing import Dict, Optional
import numpy as np
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveEngine")

class LivePredictiveEngine:
    """Computes real-time win probabilities, in-game spreads, and live totals."""

    @staticmethod
    def calculate_expected_points(down: int, distance: int, yardline_from_goal: int) -> float:
        """Dynamic down-distance-yardline valuation with bounded clamps."""
        clamped_yardline = np.clip(yardline_from_goal, 1, 99)
        base_ep = (100 - clamped_yardline) * 0.06 - 1.5
        down_penalty = (max(down, 1) - 1) * 0.75 + (max(distance, 1) * 0.05)
        return float(np.clip(base_ep - down_penalty, -2.5, 6.0))

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
        """Calculates live home win probability with zero-second boundary checks."""
        # Completed game boundary
        if seconds_remaining <= 0:
            if margin > 0:
                return {"home_win_prob": 1.0, "away_win_prob": 0.0, "fair_live_margin": float(margin), "live_spread_projection": float(-margin)}
            elif margin < 0:
                return {"home_win_prob": 0.0, "away_win_prob": 1.0, "fair_live_margin": float(margin), "live_spread_projection": float(-margin)}
            else:
                return {"home_win_prob": 0.5, "away_win_prob": 0.5, "fair_live_margin": 0.0, "live_spread_projection": 0.0}

        t_fraction = min(max(seconds_remaining, 1) / 3600.0, 1.0)
        expected_drift = pregame_spread * t_fraction

        ep = cls.calculate_expected_points(down, distance, yardline_from_goal)
        possession_adjustment = ep if (possession_team and possession_team == home_team) else -ep

        fair_live_margin = margin + expected_drift + (possession_adjustment * 0.25)
        sigma_t = max(13.5 * np.sqrt(t_fraction), 0.75)

        z = fair_live_margin / sigma_t
        home_wp = float(np.clip(norm.cdf(z), 0.001, 0.999))

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
        """Projects live total points with pace and clock runoff checks."""
        if seconds_remaining <= 0:
            return float(current_total)

        t_fraction = min(max(seconds_remaining, 0) / 3600.0, 1.0)
        remaining_expected_points = pregame_total * t_fraction
        current_drive_ep = max(0.0, (100 - np.clip(yardline_from_goal, 1, 99)) * 0.04)

        return round(float(current_total + remaining_expected_points + current_drive_ep), 1)
