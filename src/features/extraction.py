import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert, welch
from scipy.stats import kurtosis, skew


def sliding_windows(signal: np.ndarray, window_size: int, overlap: float) -> np.ndarray:
    step = int(window_size * (1 - overlap))
    return sliding_window_view(signal, window_size)[::step].copy()


def time_features(w: np.ndarray) -> dict:
    rms = np.sqrt(np.mean(w ** 2))
    mean_abs = np.mean(np.abs(w))
    peak = np.max(np.abs(w))

    return {
        "rms": rms,
        "mean": np.mean(w),
        "std": np.std(w),
        "kurtosis": kurtosis(w, bias=False),
        "skewness": skew(w, bias=False),
        "peak": peak,
        "crest_factor": peak / (rms + 1e-10),
        "peak_to_peak": np.ptp(w),
        "shape_factor": rms / (mean_abs + 1e-10),
    }


def frequency_features(w: np.ndarray, fs: float) -> dict:
    welch_freqs, power_spectral_density = welch(w, fs=fs, nperseg=len(w) // 4)
    rfft_freqs = np.fft.rfftfreq(len(w), d=1.0 / fs)
    rfft_magnitude = np.abs(np.fft.rfft(w))
    band_magnitudes = np.array_split(rfft_magnitude, 4)

    return {
        "spectral_mean": np.mean(power_spectral_density),
        "spectral_std": np.std(power_spectral_density),
        "spectral_kurtosis": kurtosis(power_spectral_density, bias=False),
        "dominant_freq": rfft_freqs[np.argmax(rfft_magnitude)],
        "total_power": np.trapezoid(power_spectral_density, welch_freqs),
        "band_1_energy": np.sum(band_magnitudes[0] ** 2),
        "band_2_energy": np.sum(band_magnitudes[1] ** 2),
        "band_3_energy": np.sum(band_magnitudes[2] ** 2),
        "band_4_energy": np.sum(band_magnitudes[3] ** 2),
    }


def envelope_features(w: np.ndarray) -> dict:
    envelope = np.abs(hilbert(w))

    return {
        "env_mean": np.mean(envelope),
        "env_std": np.std(envelope),
        "env_kurtosis": kurtosis(envelope, bias=False),
        "env_rms": np.sqrt(np.mean(envelope ** 2)),
    }


def extract_window_features(w: np.ndarray, fs: float) -> dict:
    window_features = {}
    window_features.update(time_features(w))
    window_features.update(frequency_features(w, fs))
    window_features.update(envelope_features(w))
    return window_features


def extract_bearing_features(
    df: pd.DataFrame,
    fs: float,
    window_size: int,
    overlap: float,
) -> pd.DataFrame:
    horizontal_windows = []
    vertical_windows = []
    file_indices = []
    window_indices = []

    for file_idx, snapshot_frame in df.groupby("file_idx", sort=True):
        snapshot_horizontal = sliding_windows(snapshot_frame["acc_h"].to_numpy(), window_size, overlap)
        snapshot_vertical = sliding_windows(snapshot_frame["acc_v"].to_numpy(), window_size, overlap)
        horizontal_windows.append(snapshot_horizontal)
        vertical_windows.append(snapshot_vertical)
        file_indices.extend([file_idx] * len(snapshot_horizontal))
        window_indices.extend(range(len(snapshot_horizontal)))

    channel_windows = {
        "ch_h": np.vstack(horizontal_windows),
        "ch_v": np.vstack(vertical_windows),
    }
    channel_frames = []

    for channel_prefix, windows in channel_windows.items():
        rms = np.sqrt(np.mean(windows ** 2, axis=1))
        mean_abs = np.mean(np.abs(windows), axis=1)
        peak = np.max(np.abs(windows), axis=1)
        welch_freqs, power_spectral_density = welch(
            windows,
            fs=fs,
            nperseg=window_size // 4,
            axis=1,
        )
        rfft_freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)
        rfft_magnitude = np.abs(np.fft.rfft(windows, axis=1))
        band_magnitudes = np.array_split(rfft_magnitude, 4, axis=1)
        envelope = np.abs(hilbert(windows, axis=1))

        channel_frame = pd.DataFrame(
            {
                f"{channel_prefix}_rms": rms,
                f"{channel_prefix}_mean": np.mean(windows, axis=1),
                f"{channel_prefix}_std": np.std(windows, axis=1),
                f"{channel_prefix}_kurtosis": kurtosis(windows, axis=1, bias=False),
                f"{channel_prefix}_skewness": skew(windows, axis=1, bias=False),
                f"{channel_prefix}_peak": peak,
                f"{channel_prefix}_crest_factor": peak / (rms + 1e-10),
                f"{channel_prefix}_peak_to_peak": np.ptp(windows, axis=1),
                f"{channel_prefix}_shape_factor": rms / (mean_abs + 1e-10),
                f"{channel_prefix}_spectral_mean": np.mean(power_spectral_density, axis=1),
                f"{channel_prefix}_spectral_std": np.std(power_spectral_density, axis=1),
                f"{channel_prefix}_spectral_kurtosis": kurtosis(power_spectral_density, axis=1, bias=False),
                f"{channel_prefix}_dominant_freq": rfft_freqs[np.argmax(rfft_magnitude, axis=1)],
                f"{channel_prefix}_total_power": np.trapezoid(power_spectral_density, welch_freqs, axis=1),
                f"{channel_prefix}_band_1_energy": np.sum(band_magnitudes[0] ** 2, axis=1),
                f"{channel_prefix}_band_2_energy": np.sum(band_magnitudes[1] ** 2, axis=1),
                f"{channel_prefix}_band_3_energy": np.sum(band_magnitudes[2] ** 2, axis=1),
                f"{channel_prefix}_band_4_energy": np.sum(band_magnitudes[3] ** 2, axis=1),
                f"{channel_prefix}_env_mean": np.mean(envelope, axis=1),
                f"{channel_prefix}_env_std": np.std(envelope, axis=1),
                f"{channel_prefix}_env_kurtosis": kurtosis(envelope, axis=1, bias=False),
                f"{channel_prefix}_env_rms": np.sqrt(np.mean(envelope ** 2, axis=1)),
            }
        )
        channel_frames.append(channel_frame)

    feature_frame = pd.concat(channel_frames, axis=1)
    feature_frame["file_idx"] = file_indices
    feature_frame["window_idx"] = window_indices
    return feature_frame
