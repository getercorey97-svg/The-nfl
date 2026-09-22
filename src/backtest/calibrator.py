import logging
from typing import Dict, Optional
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from src.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """Calibrates raw predictive probabilities to match true empirical frequencies."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrators: Dict[str, object] = {}

    def fit(self, target_name: str, uncalibrated_probs: np.ndarray, true_labels: np.ndarray):
        """Fits calibration curves on out-of-fold probability estimates."""
        p = np.clip(np.asarray(uncalibrated_probs, dtype=float), 1e-6, 1.0 - 1e-6)
        y = np.asarray(true_labels, dtype=int)

        if self.method == "isotonic":
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(p, y)
        elif self.method == "sigmoid":
            calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            calibrator.fit(p.reshape(-1, 1), y)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")

        self.calibrators[target_name] = calibrator
        logger.info("Fitted %s calibrator for target: %s", self.method, target_name)

    def transform(self, target_name: str, uncalibrated_probs: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated probabilities using the fitted calibrator."""
        if target_name not in self.calibrators:
            return np.clip(uncalibrated_probs, 0.0, 1.0)

        p = np.clip(np.asarray(uncalibrated_probs, dtype=float), 1e-6, 1.0 - 1e-6)
        calibrator = self.calibrators[target_name]

        if self.method == "isotonic":
            return calibrator.predict(p)
        elif self.method == "sigmoid":
            return calibrator.predict_proba(p.reshape(-1, 1))[:, 1]

    def save(self, filepath: Optional[str] = None):
        target_path = filepath or (MODEL_DIR / f"calibrator_{self.method}.joblib")
        joblib.dump(self, target_path)
        logger.info("Saved calibrator to: %s", target_path)

    @staticmethod
    def load(filepath: Optional[str] = None, method: str = "isotonic") -> "ProbabilityCalibrator":
        target_path = filepath or (MODEL_DIR / f"calibrator_{method}.joblib")
        return joblib.load(target_path)
