"""
ECG loading, preprocessing, and demo synthesis.

Preprocessing matches the IDSC 2026 Colab notebook exactly:
  - High-pass Butterworth order 4 at 0.5 Hz
  - Low-pass Butterworth order 4 at 40 Hz
  - No 60 Hz notch (Nyquist = 50 Hz at fs=100)
"""

import numpy as np
from scipy.signal import butter, filtfilt

LEAD_NAMES_12 = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


class ECGProcessor:
    def load_wfdb(self, record_path):
        import wfdb

        rec = wfdb.rdrecord(record_path)
        signals = rec.p_signal
        meta = {
            "fs": rec.fs,
            "n_sig": rec.n_sig,
            "sig_len": rec.sig_len,
            "lead_names": (
                rec.sig_name
                if rec.sig_name
                else LEAD_NAMES_12[: rec.n_sig]
            ),
            "units": rec.units,
        }
        return signals, meta

    # ── demo ECG synthesis ─────────────────────────────────
    def generate_demo_ecg(self, duration=12, fs=100, brugada=None):
        """Match Brugada-HUCA: 12 seconds, 100 Hz, 12 leads."""
        if brugada is None:
            brugada = bool(np.random.choice([True, False]))

        n = duration * fs
        t = np.linspace(0, duration, n)
        signals = np.zeros((n, 12))

        for i in range(12):
            signals[:, i] = self._synth_lead(t, fs, i, brugada)

        meta = {
            "fs": fs,
            "n_sig": 12,
            "sig_len": n,
            "lead_names": list(LEAD_NAMES_12),
            "units": ["mV"] * 12,
            "is_brugada": brugada,
        }
        return signals, meta

    @staticmethod
    def _gaussian(t, amp, center, width):
        return amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    def _synth_lead(self, t, fs, idx, brugada):
        hr = 72 + np.random.uniform(-5, 5)
        period = 60.0 / hr
        ecg = np.zeros_like(t)
        amp_scale = 1.0 + 0.12 * np.sin(idx * 0.5)

        for bs in np.arange(0, t[-1], period):
            bt = t - bs
            m = (bt >= 0) & (bt < period)

            ecg[m] += self._gaussian(bt[m], 0.15 * amp_scale, 0.10, 0.040)
            ecg[m] += self._gaussian(bt[m], -0.20, 0.18, 0.005)
            ecg[m] += self._gaussian(bt[m], 1.00 * amp_scale, 0.20, 0.008)
            ecg[m] += self._gaussian(bt[m], -0.30, 0.22, 0.005)

            if brugada and idx in (6, 7):
                ecg[m] += self._gaussian(bt[m], 0.45, 0.28, 0.025)
                ecg[m] += self._gaussian(bt[m], -0.35, 0.40, 0.055)
            elif brugada and idx == 8:
                ecg[m] += self._gaussian(bt[m], 0.20, 0.28, 0.025)
                ecg[m] += self._gaussian(bt[m], -0.15, 0.40, 0.055)
            else:
                ecg[m] += self._gaussian(
                    bt[m], 0.30 * amp_scale, 0.36, 0.060
                )

        ecg += 0.02 * np.random.randn(len(t))
        ecg += 0.06 * np.sin(
            2 * np.pi * 0.15 * t + np.random.rand() * 6.28
        )
        ecg += 0.03 * np.sin(
            2 * np.pi * 0.35 * t + np.random.rand() * 6.28
        )
        return ecg

    # ── preprocessing (matches Colab Step 6 EXACTLY) ───────
    def preprocess(self, signals, fs):
        """
        Pipeline from Colab:
          1. High-pass at 0.5 Hz (butter order 4) → baseline wander
          2. Low-pass at 40 Hz (butter order 4) → HF noise
          NO 60 Hz notch — impossible at fs=100 Hz
        """
        out = np.zeros_like(signals)
        for i in range(signals.shape[1]):
            out[:, i] = self._filter(signals[:, i], fs)
        return out

    @staticmethod
    def _filter(x, fs):
        nyq = fs / 2.0

        # 1. baseline_wander_removal: butter(4, 0.5/nyq, 'high')
        if nyq > 0.5:
            b, a = butter(4, 0.5 / nyq, btype="high")
            x = filtfilt(b, a, x)

        # 2. lowpass_cleanup: butter(4, 40/nyq, 'low') with clamp
        if nyq > 40:
            norm_cutoff = 40.0 / nyq
            norm_cutoff = min(norm_cutoff, 0.99)  # Colab clamp
            b, a = butter(4, norm_cutoff, btype="low")
            x = filtfilt(b, a, x)

        # NO 60 Hz notch
        return x