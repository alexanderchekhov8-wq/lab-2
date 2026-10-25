import numpy as np


def compute_errors(s_num: np.ndarray, s_exact: np.ndarray) -> dict:
    """Считает нормы L1, L2, Linf разности численного и точного решений."""
    diff = s_num - s_exact
    return {
        "L1": float(np.mean(np.abs(diff))),
        "L2": float(np.sqrt(np.mean(diff ** 2))),
        "Linf": float(np.max(np.abs(diff))),
    }


def compute_fwhm(x: np.ndarray, s: np.ndarray) -> float:
    """Ширина пика на полувысоте (FWHM). Предполагается один пик."""
    s_max = np.max(s)
    if s_max <= 0:
        return 0.0
    half = s_max / 2.0
    above = np.where(s >= half)[0]
    if len(above) < 2:
        return 0.0
    return float(x[above[-1]] - x[above[0]])