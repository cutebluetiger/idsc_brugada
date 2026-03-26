"""
Feature extractor — matches IDSC 2026 Colab notebook EXACTLY.

Colab functions replicated:
  smoothed_spectrogram()    → Step 7
  marginal_energy_features() → Step 8
  nonlinear_features()       → Step 9
  extract_all_features()     → Step 10
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import stft
from scipy.ndimage import gaussian_filter
import antropy as ant  # type: ignore

LEAD_NAMES_12 = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# ── leads used in training (Step 10 of Colab) ──
MODEL_LEADS = ["V1", "V2", "V3", "II", "V4"]

# ── frequency bands (Step 8 of Colab) ──
ECG_FREQ_BANDS = {
    "low_pt": (0.5, 5.0),
    "mid_qrs": (5.0, 15.0),
    "high_qrs": (15.0, 30.0),
    "upper": (30.0, 40.0),
}

# ── exact 25 features the XGBoost model expects ──
MODEL_FEATURES = [
    "V1_perm_entropy",
    "V2_E_mid_qrs_norm",
    "II_hjorth_mobility",
    "V4_E_mid_qrs_norm",
    "V4_E_upper",
    "V1_skewness",
    "V1_hjorth_mobility",
    "V1_hjorth_complexity",
    "II_kurtosis",
    "V4_E_upper_norm",
    "V3_E_mid_qrs_norm",
    "V2_hjorth_mobility",
    "V3_E_upper_norm",
    "V2_zero_crossing_rate",
    "V1_kurtosis",
    "II_E_upper_norm",
    "V1_zero_crossing_rate",
    "II_E_low_pt",
    "V1_E_high_qrs_norm",
    "II_p2p",
    "V1_E_mid_qrs_norm",
    "V4_E_low_pt_norm",
    "V3_kurtosis",
    "V4_skewness",
    "II_E_low_pt_norm",
]


# ══════════════════════════════════════════════════════════════
# EXACT COPIES OF COLAB FUNCTIONS
# ══════════════════════════════════════════════════════════════

def smoothed_spectrogram(signal_1d, fs=100.0, sigma=1.0, n_freqs=64):
    """
    Colab Step 7 — EXACT COPY.

    Gaussian-smoothed STFT spectrogram.
    Window: ('gaussian', 2.5)
    nperseg: min(n_freqs * 2, n) = min(128, n)
    noverlap: 75%
    """
    n = len(signal_1d)
    nperseg = min(n_freqs * 2, n)
    noverlap = int(nperseg * 0.75)

    freqs, times, Zxx = stft(
        signal_1d,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=("gaussian", 2.5),
    )

    power = np.abs(Zxx) ** 2
    power_smooth = gaussian_filter(power, sigma=sigma)

    return power_smooth, freqs, times


def marginal_energy_features(tfd, freqs, bands):
    """
    Colab Step 8 — EXACT COPY.

    Band-specific marginal energy from a time-frequency distribution.
    Also computes spectral entropy of the frequency marginal.
    """
    features = {}
    total_energy = tfd.sum() + 1e-12

    for band_name, (flo, fhi) in bands.items():
        mask = (freqs >= flo) & (freqs < fhi)
        band_energy = tfd[mask, :].sum()
        features[f"E_{band_name}"] = band_energy
        features[f"E_{band_name}_norm"] = band_energy / total_energy

    # Spectral entropy of the frequency marginal
    marginal = tfd.sum(axis=1)
    marginal_norm = marginal / (marginal.sum() + 1e-12)
    features["spectral_entropy"] = -np.sum(
        marginal_norm * np.log2(marginal_norm + 1e-12)
    )

    return features


def nonlinear_features(signal_1d):
    """
    Colab Step 9 — EXACT COPY.

    Nonlinear and statistical features from a single ECG lead.
    Uses antropy for permutation entropy and sample entropy.
    """
    feats = {}

    # ── Time-domain statistics ──
    feats["mean"] = np.mean(signal_1d)
    feats["std"] = np.std(signal_1d)
    feats["skewness"] = float(skew(signal_1d))
    feats["kurtosis"] = float(kurtosis(signal_1d))
    feats["rms"] = np.sqrt(np.mean(signal_1d ** 2))
    feats["p2p"] = signal_1d.max() - signal_1d.min()

    # ── Signal energy and power ──
    feats["energy"] = np.sum(signal_1d ** 2)
    feats["power"] = np.mean(signal_1d ** 2)

    # ── Hjorth parameters ──
    diff1 = np.diff(signal_1d)
    diff2 = np.diff(diff1)
    var0 = np.var(signal_1d) + 1e-12
    var1 = np.var(diff1) + 1e-12
    var2 = np.var(diff2) + 1e-12
    feats["hjorth_mobility"] = np.sqrt(var1 / var0)
    feats["hjorth_complexity"] = (
        np.sqrt(var2 / var1) / feats["hjorth_mobility"]
    )

    # ── Sample Entropy (antropy) ──
    try:
        feats["sample_entropy"] = ant.sample_entropy(signal_1d)
    except Exception:
        feats["sample_entropy"] = np.nan

    # ── Permutation Entropy (antropy, normalised) ──
    try:
        feats["perm_entropy"] = ant.perm_entropy(
            signal_1d, normalize=True
        )
    except Exception:
        feats["perm_entropy"] = np.nan

    # ── Zero Crossing Rate ──
    feats["zero_crossing_rate"] = (
        np.sum(np.diff(np.sign(signal_1d)) != 0) / len(signal_1d)
    )

    return feats


# ══════════════════════════════════════════════════════════════
# MAIN EXTRACTOR CLASS
# ══════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Extract the exact 25 features for the Brugada XGBoost model."""

    def extract(self, signals, fs):
        """
        Replicates Colab Step 10: extract_all_features()

        Parameters
        ----------
        signals : ndarray (n_samples, n_leads)
        fs      : float  sampling frequency

        Returns
        -------
        model_features : ndarray (1, 25)
        model_names    : list[str]  length-25
        all_computed   : dict[str, float]  every feature for dashboard
        """
        # Pad to 12 leads if needed
        n_leads = signals.shape[1]
        if n_leads < 12:
            pad = np.zeros((signals.shape[0], 12 - n_leads))
            signals = np.concatenate([signals, pad], axis=1)
            print(f"⚠️  Padded {n_leads} → 12 leads")

        # Compute all features for each model lead
        # (matches Colab loop in extract_all_features)
        all_computed = {}
        for lead_name in MODEL_LEADS:
            idx = LEAD_NAMES_12.index(lead_name)
            lead_sig = signals[:, idx]
            prefix = lead_name.replace("-", "neg").replace(" ", "_")

            # Smoothed spectrogram features (Step 7 + 8)
            tfd, freqs, _ = smoothed_spectrogram(
                lead_sig, fs=fs, sigma=1.0
            )
            e_feats = marginal_energy_features(
                tfd, freqs, ECG_FREQ_BANDS
            )
            for k, v in e_feats.items():
                all_computed[f"{prefix}_{k}"] = float(v)

            # Nonlinear / statistical features (Step 9)
            nl_feats = nonlinear_features(lead_sig)
            for k, v in nl_feats.items():
                all_computed[f"{prefix}_{k}"] = float(v)

        # Replace NaN with 0 (matches Colab median imputation fallback)
        for k in all_computed:
            if not np.isfinite(all_computed[k]):
                all_computed[k] = 0.0

        # Pick the 25 model features in exact order
        model_values = []
        for fname in MODEL_FEATURES:
            val = all_computed.get(fname, 0.0)
            model_values.append(val)

        return (
            np.array([model_values]),
            list(MODEL_FEATURES),
            all_computed,
        )