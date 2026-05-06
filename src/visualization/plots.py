from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_raw_signal(df: pd.DataFrame, channel: str, save_path: Path):
    mean_amplitude = df.groupby("file_idx")[channel].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mean_amplitude.index, mean_amplitude.values, linewidth=1.2, label=channel)
    ax.set_xlabel("File index (× 10s)")
    ax.set_ylabel("Mean amplitude")
    ax.set_title(f"Mean {channel} Amplitude Over Bearing Lifetime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_rolling_rms(
    rms_series: np.ndarray,
    save_path: Path,
    threshold: float = None,
    failure_idx: int = None,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(rms_series)), rms_series, linewidth=1.2, label="RMS")
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="Threshold")
    if failure_idx is not None:
        ax.axvline(failure_idx, color="grey", linestyle="--", linewidth=1.2, label="Failure")
    ax.set_xlabel("File index (× 10s)")
    ax.set_ylabel("RMS acceleration")
    ax.set_title("Rolling RMS Over Bearing Lifetime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_anomaly_scores(
    scores: np.ndarray,
    label: str,
    save_path: Path,
    threshold: float = None,
    failure_idx: int = None,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(scores)), scores, linewidth=1.2, label=label)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="Threshold")
    if failure_idx is not None:
        ax.axvline(failure_idx, color="grey", linestyle="--", linewidth=1.2, label="Failure")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"{label} Anomaly Score Timeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fft_comparison(
    early_window: np.ndarray,
    late_window: np.ndarray,
    fs: float,
    save_path: Path,
):
    early_freqs = np.fft.rfftfreq(len(early_window), d=1.0 / fs)
    late_freqs = np.fft.rfftfreq(len(late_window), d=1.0 / fs)
    early_magnitude = np.abs(np.fft.rfft(early_window))
    late_magnitude = np.abs(np.fft.rfft(late_window))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    axes[0].plot(early_freqs, early_magnitude, linewidth=1.0, label="Healthy")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("FFT magnitude")
    axes[0].set_title("Early-Life Window Spectrum")
    axes[0].legend()

    axes[1].plot(late_freqs, late_magnitude, linewidth=1.0, color="tab:red", label="Degraded")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_title("Late-Life Window Spectrum")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reconstruction_error(
    errors: np.ndarray,
    save_path: Path,
    threshold: float = None,
    failure_idx: int = None,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(errors)), errors, linewidth=1.2, label="Reconstruction MSE")
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="Threshold")
    if failure_idx is not None:
        ax.axvline(failure_idx, color="grey", linestyle="--", linewidth=1.2, label="Failure")
    ax.set_xlabel("Sequence index")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("LSTM Autoencoder Reconstruction Error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(results: dict[str, tuple], save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, (fpr_array, tpr_array, auc_score) in results.items():
        ax.plot(fpr_array, tpr_array, linewidth=1.4, label=f"{model_name} AUC={auc_score:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(results: dict[str, tuple], save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, (precision_array, recall_array, auc_score) in results.items():
        ax.plot(recall_array, precision_array, linewidth=1.4, label=f"{model_name} AUC={auc_score:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison_table(metrics_df: pd.DataFrame, save_path: Path):
    fig_height = max(2.5, 0.45 * (len(metrics_df) + 1))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=metrics_df.round(4).astype(str).values,
        colLabels=metrics_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    ax.set_title("Model Comparison Metrics", pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
