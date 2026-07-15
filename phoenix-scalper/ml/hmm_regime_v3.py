import numpy as np
import logging
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


class RegimeDetector:
    n_states = 3

    def __init__(self, n_states=3, n_iter=100):
        self.n_states = n_states
        self.n_iter = n_iter
        self._init_params()

    def _init_params(self):
        self.pi = np.array([0.3, 0.4, 0.3], dtype=np.float64)

        self.A = np.array([
            [0.85, 0.10, 0.05],
            [0.20, 0.60, 0.20],
            [0.05, 0.10, 0.85]
        ], dtype=np.float64)

        self.mu = np.array([
            [0.001, 0.02, 0.3],
            [0.0, 0.05, 0.5],
            [-0.001, 0.02, 0.5]
        ], dtype=np.float64)

        self.Sigma = np.array([
            [[0.0001, 0, 0], [0, 0.0004, 0], [0, 0, 0.4]],
            [[0.0001, 0, 0], [0, 0.0009, 0], [0, 0, 0.9]],
            [[0.0001, 0, 0], [0, 0.0004, 0], [0, 0, 0.4]]
        ], dtype=np.float64)

    def fit(self, observations: np.ndarray):
        if len(observations) < 10:
            logger.warning("Not enough observations for HMM fitting")
            return
        self._init_params()
        for iteration in range(self.n_iter):
            log_lik, gamma, xi = self._forward_backward(observations)
            self._m_step(observations, gamma, xi)
            if iteration % 20 == 0:
                logger.info(f"HMM iter {iteration}: log-likelihood = {log_lik:.4f}")

    def predict_regime(self, observations: np.ndarray) -> np.ndarray:
        return self._viterbi(observations)

    def predict_regime_probs(self, observations: np.ndarray) -> np.ndarray:
        _, gamma, _ = self._forward_backward(observations)
        return gamma

    def predict_regime_fast(self, observations: np.ndarray) -> np.ndarray:
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        observations = np.nan_to_num(observations, nan=0.0, posinf=10.0, neginf=-10.0)
        B = self._emission_prob(observations)
        B = np.nan_to_num(B, nan=1e-10)
        B = np.clip(B, 1e-10, None)
        weighted = B * self.pi[np.newaxis, :]
        return np.argmax(weighted, axis=1)

    def predict_regime_probs_fast(self, observations: np.ndarray) -> np.ndarray:
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        observations = np.nan_to_num(observations, nan=0.0, posinf=10.0, neginf=-10.0)
        B = self._emission_prob(observations)
        B = np.nan_to_num(B, nan=1e-10)
        B = np.clip(B, 1e-10, None)
        weighted = B * self.pi[np.newaxis, :]
        total = np.sum(weighted, axis=1, keepdims=True)
        total = np.clip(total, 1e-10, None)
        return weighted / total

    def _forward_backward(self, obs):
        T = len(obs)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        log_A = np.log(self.A + 1e-10)
        emission = self._emission_prob(obs)
        emission = np.nan_to_num(emission, nan=-100.0, posinf=-100.0, neginf=-100.0)
        log_B = np.log(np.clip(emission, 1e-10, None))
        log_pi = np.log(self.pi + 1e-10)

        log_alpha = np.zeros((T, self.n_states))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_A + log_alpha[t-1, :, np.newaxis], axis=0) + log_B[t]

        log_beta = np.zeros((T, self.n_states))
        for t in range(T-2, -1, -1):
            log_beta[t] = logsumexp(log_A.T + log_B[t+1, np.newaxis, :] + log_beta[t+1, np.newaxis, :], axis=1)

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        gamma = np.nan_to_num(gamma, nan=1.0/self.n_states)

        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            log_xi_t = log_alpha[t, :, np.newaxis] + log_A + log_B[t+1, np.newaxis, :] + log_beta[t+1, np.newaxis, :]
            log_xi_t -= logsumexp(log_xi_t)
            xi[t] = np.exp(log_xi_t)
        xi = np.nan_to_num(xi, nan=0.0)

        log_lik = logsumexp(log_alpha[-1])
        return log_lik, gamma, xi

    def _viterbi(self, obs):
        T = len(obs)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)

        log_A = np.log(self.A + 1e-10)
        log_B = np.log(self._emission_prob(obs) + 1e-10)

        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        delta[0] = np.log(self.pi + 1e-10) + log_B[0]

        for t in range(1, T):
            for j in range(self.n_states):
                candidates = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[t, j]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states

    def _emission_prob(self, obs):
        T = obs.shape[0]
        d = obs.shape[1]
        log_B = np.zeros((T, self.n_states))

        for j in range(self.n_states):
            diff = obs - self.mu[j]
            try:
                L = np.linalg.cholesky(self.Sigma[j])
                L_inv = np.linalg.inv(L)
                diff_transformed = diff @ L_inv.T
                log_det = 2.0 * np.sum(np.log(np.diag(L) + 1e-10))
                log_B[:, j] = -0.5 * np.sum(diff_transformed ** 2, axis=1) - 0.5 * log_det - 0.5 * d * np.log(2 * np.pi)
            except np.linalg.LinAlgError:
                diag = np.diag(self.Sigma[j])
                log_B[:, j] = -0.5 * np.sum((diff ** 2) / (diag + 1e-10), axis=1) - 0.5 * np.sum(np.log(np.abs(diag) + 1e-10)) - 0.5 * d * np.log(2 * np.pi)

        log_B = np.nan_to_num(log_B, nan=-100.0, posinf=-100.0, neginf=-100.0)
        log_B = np.clip(log_B, -100.0, 0.0)
        return np.exp(log_B)

    def _m_step(self, obs, gamma, xi):
        T = obs.shape[0]
        self.pi = gamma[0].copy()

        gamma_sum = np.sum(xi, axis=(0, 2))
        self.A = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                denom = gamma_sum[i] if gamma_sum[i] > 0 else 1.0
                self.A[i, j] = np.sum(xi[:, i, j]) / denom

        for j in range(self.n_states):
            gamma_j = gamma[:, j]
            self.mu[j] = np.average(obs, axis=0, weights=gamma_j)
            diff = obs - self.mu[j]
            self.Sigma[j] = np.zeros((obs.shape[1], obs.shape[1]))
            for t in range(T):
                self.Sigma[j] += gamma_j[t] * np.outer(diff[t], diff[t])
            self.Sigma[j] += np.eye(obs.shape[1]) * 1e-6


def compute_hmm_features(
    returns: np.ndarray,
    vol: np.ndarray,
    volume_change: np.ndarray,
    full_returns: np.ndarray = None,
    full_vol: np.ndarray = None,
    full_volume_change: np.ndarray = None,
) -> dict:
    """
    Train HMM on provided data (should be historical only) and predict on full data.

    Args:
        returns: Training log returns (historical window only)
        vol: Training volatility (historical window only)
        volume_change: Training volume changes (historical window only)
        full_returns: Full series for prediction (if None, uses training data)
        full_vol: Full series for prediction
        full_volume_change: Full series for prediction

    Returns:
        dict of feature arrays aligned to full_returns length
    """
    train_n = len(returns)
    obs = np.column_stack([returns, vol, volume_change])

    max_train = min(1000, train_n)
    if train_n > max_train:
        idx = np.linspace(0, train_n - 1, max_train, dtype=int)
        obs_train = obs[idx]
    else:
        obs_train = obs

    hmm = RegimeDetector(n_states=3, n_iter=5)
    hmm.fit(obs_train)

    if full_returns is not None and full_vol is not None and full_volume_change is not None:
        predict_obs = np.column_stack([full_returns, full_vol, full_volume_change])
        n = len(full_returns)
    else:
        predict_obs = obs
        n = train_n

    regime = hmm.predict_regime_fast(predict_obs)
    regime_probs = hmm.predict_regime_probs_fast(predict_obs)

    regime_stability = np.zeros(n)
    transition_risk = np.zeros(n)
    vol_regime = np.zeros(n)
    trend_strength = np.zeros(n)

    vol_col = 1
    mu_col = 1  # match V2: uses mu_col for volatility normalization

    for i in range(n):
        regime_stability[i] = 1.0 - np.sum(regime_probs[i] ** 2)

        if regime[i] == 0:
            transition_risk[i] = hmm.A[0, 1] + hmm.A[0, 2]
        elif regime[i] == 1:
            transition_risk[i] = hmm.A[1, 0] + hmm.A[1, 2]
        else:
            transition_risk[i] = hmm.A[2, 0] + hmm.A[2, 1]

        vol_regime[i] = predict_obs[i, vol_col] / (abs(hmm.mu[regime[i], mu_col]) + 1e-10)
        trend_strength[i] = abs(hmm.mu[regime[i], 0]) / (abs(hmm.Sigma[regime[i], mu_col, mu_col]) + 1e-10)

    return {
        "hmm_regime": regime,
        "hmm_p_bull": regime_probs[:, 0],
        "hmm_p_range": regime_probs[:, 1],
        "hmm_p_bear": regime_probs[:, 2],
        "hmm_regime_stability": regime_stability,
        "hmm_transition_risk": transition_risk,
        "hmm_vol_regime": vol_regime,
        "hmm_trend_strength": trend_strength,
    }
