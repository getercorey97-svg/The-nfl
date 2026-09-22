import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class BacktestEvaluator:
    """Computes empirical validation metrics, financial ROI, and Kelly betting stakes."""

    @staticmethod
    def american_to_decimal(american_odds: float) -> float:
        """Converts American odds (+150, -110) to European decimal odds."""
        if american_odds > 0:
            return (american_odds / 100.0) + 1.0
        else:
            return (100.0 / abs(american_odds)) + 1.0

    @staticmethod
    def decimal_to_prob(decimal_odds: float) -> float:
        """Converts decimal odds to implied break-even probability."""
        if decimal_odds <= 1.0:
            return 1.0
        return 1.0 / decimal_odds

    @classmethod
    def calculate_ev(cls, true_prob: float, american_odds: float) -> float:
        """Calculates expected percentage return per unit staked: EV = (P * b) - (1 - P)."""
        b = cls.american_to_decimal(american_odds) - 1.0
        ev = (true_prob * b) - (1.0 - true_prob)
        return float(ev)

    @classmethod
    def calculate_kelly_fraction(cls, true_prob: float, american_odds: float, fraction: float = 0.25) -> float:
        """Calculates fractional Kelly stake size (default Quarter-Kelly for risk mitigation)."""
        b = cls.american_to_decimal(american_odds) - 1.0
        if b <= 0:
            return 0.0
        full_kelly = (b * true_prob - (1.0 - true_prob)) / b
        if full_kelly <= 0:
            return 0.0
        return float(full_kelly * fraction)

    @staticmethod
    def compute_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculates Expected Calibration Error (ECE)."""
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(probs)
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(labels[mask])
                bin_conf = np.mean(probs[mask])
                ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
        return float(ece)

    @classmethod
    def audit_betting_performance(cls, bets_df: pd.DataFrame, default_odds: float = -110.0) -> Dict[str, float]:
        """Audits empirical profit, yield, win rate, and drawdown across settled wagers."""
        if bets_df.empty:
            return {"total_bets": 0, "roi_pct": 0.0, "net_units": 0.0, "win_rate_pct": 0.0}

        df = bets_df.copy()
        if "odds" not in df.columns:
            df["odds"] = default_odds

        df["dec_odds"] = df["odds"].apply(cls.american_to_decimal)

        # Standard 1-unit flat stake evaluation
        df["payout"] = np.where(df["won"] == 1, df["dec_odds"] - 1.0, -1.0)
        if "push" in df.columns:
            df.loc[df["push"] == 1, "payout"] = 0.0

        total_bets = len(df)
        total_profit = float(df["payout"].sum())
        roi = float((total_profit / total_bets) * 100.0) if total_bets > 0 else 0.0
        win_rate = float(df["won"].mean() * 100.0)

        return {
            "total_bets": total_bets,
            "net_units": round(total_profit, 2),
            "roi_pct": round(roi, 2),
            "win_rate_pct": round(win_rate, 2)
        }
