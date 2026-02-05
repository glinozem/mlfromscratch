from __future__ import annotations

import numpy as np


def check_X(X: object) -> np.ndarray:
    Xv = np.asarray(X, dtype=float)
    if Xv.ndim != 2:
        raise ValueError("X must be 2D array-like: shape (n_samples, n_features)")
    if Xv.shape[0] == 0:
        raise ValueError("X must have at least one sample")
    return Xv


def check_y(y: object, n_samples: int) -> np.ndarray:
    yv = np.asarray(y, dtype=float).reshape(-1)
    if yv.shape[0] != n_samples:
        raise ValueError("y length must match X.shape[0]")
    return yv


def check_X_y(X: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    Xv = check_X(X)
    yv = check_y(y, Xv.shape[0])
    return Xv, yv
