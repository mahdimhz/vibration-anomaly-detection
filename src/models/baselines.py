import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import SEED


def rolling_rms_scores(
    rms_series: np.ndarray,
    train_ratio: float,
    sigma: float,
) -> tuple[np.ndarray, float]:
    train_count = int(len(rms_series) * train_ratio)
    train_rms = rms_series[:train_count]
    threshold = np.mean(train_rms) + sigma * np.std(train_rms)
    return rms_series, threshold


def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float,
) -> tuple[IsolationForest, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = IsolationForest(contamination=contamination, random_state=SEED)
    clf.fit(X_train_scaled)
    return clf, scaler


def score_isolation_forest(clf, scaler, X: np.ndarray) -> np.ndarray:
    return -clf.decision_function(scaler.transform(X))


def fit_one_class_svm(
    X_train: np.ndarray,
    nu: float,
) -> tuple[OneClassSVM, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = OneClassSVM(nu=nu)
    clf.fit(X_train_scaled)
    return clf, scaler


def score_one_class_svm(clf, scaler, X: np.ndarray) -> np.ndarray:
    return -clf.decision_function(scaler.transform(X))
