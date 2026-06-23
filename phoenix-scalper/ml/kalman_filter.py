import numpy as np
import logging

logger = logging.getLogger(__name__)


class KalmanPricePredictor:
    """
    1D Kalman filter for price with adaptive noise estimation.

    State: [price_level, price_trend]

    Key outputs:
      - kf_price: filtered (smoothed) price
      - kf_trend: estimated trend direction (+/-)
      - kf_acceleration: trend acceleration (2nd derivative)
      - kf_prediction: next-period price forecast
      - kf_confidence: prediction confidence (0-1, based on innovation covariance)
      - kf_direction: +1 (up), -1 (down), 0 (flat)
    """

    def __init__(self, process_noise=0.01, measurement_noise_factor=1.0, min_trend_threshold=0.0001):
        self.F = np.array([[1, 1], [0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0]], dtype=np.float64)
        self.x = np.array([0.0, 0.0], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self.Q_base = np.array([
            [process_noise**2, 0],
            [0, process_noise**2 * 0.1]
        ], dtype=np.float64)

        self.measurement_noise_factor = measurement_noise_factor
        self.min_trend_threshold = min_trend_threshold
        self.R = 0.01
        self.initialized = False

    def reset(self, initial_price: float = None):
        if initial_price is not None:
            self.x = np.array([initial_price, 0.0], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self.initialized = True

    def adapt_noise(self, atr_pct: float, volume_ratio: float):
        if np.isnan(atr_pct) or atr_pct <= 0:
            atr_pct = 0.01
        if np.isnan(volume_ratio) or volume_ratio <= 0:
            volume_ratio = 1.0
        R_base = atr_pct ** 2
        vol_adj = max(0.5, 1.0 / (volume_ratio + 0.1))
        self.R = R_base * vol_adj * self.measurement_noise_factor

    def update(self, price: float) -> dict:
        if np.isnan(price) or np.isinf(price):
            price = self.x[0] if self.initialized else 0.0

        if not self.initialized:
            self.reset(price)

        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q_base

        z = np.array([price], dtype=np.float64)
        y_innov = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        S_scalar = S[0, 0]

        K = P_pred @ self.H.T / (S_scalar + 1e-10)

        self.x = x_pred + K.flatten() * y_innov
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        kf_price = self.x[0]
        kf_trend = self.x[1]

        kf_prediction = (self.F @ self.x)[0]

        kf_confidence = 1.0 / (1.0 + abs(S_scalar) / (self.R + 1e-10))
        kf_confidence = np.clip(kf_confidence, 0.0, 1.0)

        if kf_trend > self.min_trend_threshold:
            kf_direction = 1
        elif kf_trend < -self.min_trend_threshold:
            kf_direction = -1
        else:
            kf_direction = 0

        return {
            "kf_price": float(kf_price),
            "kf_trend": float(kf_trend),
            "kf_prediction": float(kf_prediction),
            "kf_confidence": float(kf_confidence),
            "kf_direction": int(kf_direction),
            "kf_innovation": float(y_innov[0]),
            "kf_S": float(S_scalar),
        }


def compute_kalman_features(prices: np.ndarray, atr_pct: np.ndarray, volume_ratio: np.ndarray) -> dict:
    n = len(prices)
    kf = KalmanPricePredictor()

    kf_price = np.zeros(n)
    kf_trend = np.zeros(n)
    kf_prediction = np.zeros(n)
    kf_confidence = np.zeros(n)
    kf_direction = np.zeros(n, dtype=int)
    kf_innovation = np.zeros(n)
    kf_S = np.zeros(n)

    for i in range(n):
        kf.adapt_noise(atr_pct[i] if i < len(atr_pct) else atr_pct[-1],
                      volume_ratio[i] if i < len(volume_ratio) else volume_ratio[-1])
        result = kf.update(prices[i])

        kf_price[i] = result["kf_price"]
        kf_trend[i] = result["kf_trend"]
        kf_prediction[i] = result["kf_prediction"]
        kf_confidence[i] = result["kf_confidence"]
        kf_direction[i] = result["kf_direction"]
        kf_innovation[i] = result["kf_innovation"]
        kf_S[i] = result["kf_S"]

    kf_price_momentum = np.zeros(n)
    kf_trend_acceleration = np.zeros(n)
    kf_prediction_error = np.zeros(n)
    kf_regime_score = np.zeros(n)
    kf_vol_of_trend = np.zeros(n)
    kf_atr_ratio = np.zeros(n)

    for i in range(1, n):
        if kf_price[i-1] != 0:
            kf_price_momentum[i] = (kf_price[i] - kf_price[i-1]) / kf_price[i-1]
        kf_trend_acceleration[i] = kf_trend[i] - kf_trend[i-1]
        if kf_price[i] != 0:
            kf_prediction_error[i] = (prices[i] - kf_prediction[i]) / kf_price[i]
        kf_regime_score[i] = kf_confidence[i] * kf_direction[i]

    for i in range(20, n):
        kf_vol_of_trend[i] = np.std(kf_trend[i-20:i])

    for i in range(n):
        if abs(kf_trend[i]) > 1e-10:
            kf_atr_ratio[i] = atr_pct[i] / abs(kf_trend[i]) if i < len(atr_pct) else 0
        else:
            kf_atr_ratio[i] = 0

    return {
        "kf_price": kf_price,
        "kf_trend": kf_trend,
        "kf_prediction": kf_prediction,
        "kf_confidence": kf_confidence,
        "kf_direction": kf_direction,
        "kf_innovation": kf_innovation,
        "kf_S": kf_S,
        "kf_price_momentum": kf_price_momentum,
        "kf_trend_acceleration": kf_trend_acceleration,
        "kf_prediction_error": kf_prediction_error,
        "kf_regime_score": kf_regime_score,
        "kf_vol_of_trend": kf_vol_of_trend,
        "kf_atr_ratio": kf_atr_ratio,
    }
