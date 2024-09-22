import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = 0

    ed_dist = np.sqrt(
        np.sum(
            (ts1 - ts2) ** 2
        )
    )

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    n = len(ts1)

    norm_ed_dist = np.sqrt(
        np.abs(
            2 * n * (
                1 - (
                    (np.dot(ts1, ts2) - n * np.mean(ts1) * np.mean(ts2))
                    / (n * np.std(ts1) * np.std(ts2))
                )
            )
        )
    )

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    dtw_dist = 0

    n = len(ts1)

    d = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                d[i, j] = 0

            elif i == 0 or j == 0:
                d[i, j] = np.inf

            else:
                d[i, j] = (
                    (ts1[i - 1] - ts2[j - 1]) ** 2
                    + min(
                        d[i - 1, j],
                        d[i, j - 1],
                        d[i - 1, j - 1]
                    )
                )

    dtw_dist = d[n, n]

    return dtw_dist
