import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numba import njit

logger = logging.getLogger(__name__)


def count_calls(func):
    """Decorator: increments func.call_count each time func is called."""
    from functools import wraps
    import threading

    lock = threading.Lock()

    @wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0
    return wrapper


def merge_dicts(first, second):
    merged = first.copy()
    collision = False

    for k, v in second.items():
        if k in merged:
            collision = True
            logger.warning(
                f"Key clash on '{k}': overwriting value {merged[k]} with {v}"
            )
        merged[k] = v

    if not collision:
        logger.info(f"Safely merged dictionaries with no collisions.")

    return merged


def write_tsv(opath, df=None, header=True, index=False, index_label=None):
    if df is None:
        df = pd.DataFrame()

    logger.info(f"Writing to {opath}.")

    df.to_csv(opath, sep="\t", header=header, index=index, index_label=index_label)


def write_fig(opath, fig=None, transparent=True, bbox_inches="tight"):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    logger.info(f"Writing figure to {opath}.")
    fig.savefig(opath, format="pdf", transparent=transparent, bbox_inches=bbox_inches)


@njit
def top_hat_sum(arr, width):
    # TODO HACK?
    arr = np.atleast_2d(arr)
    
    n = arr.shape[0]
    out = np.empty(arr.shape, dtype=arr.dtype)

    left = width // 2
    right = width - left - 1

    acc = np.zeros(arr[0].shape, dtype=arr.dtype)

    for i in range(n):
        acc[...] = 0

        for k in range(-left, right + 1):
            idx = i + k
            if 0 <= idx < n:
                acc += arr[idx, ...]
        out[i, ...] = acc
    return out


def cast_clone_label(label, with_normal=False):
    num = label.replace("clone", "").strip()
    num = int(num)

    if not (-1 <= num <= 3999):
        raise ValueError("Input must be an integer between -1 and 3999.")

    if num == -1:
        return "WARN"
    elif num == 0:
        if with_normal:
            return "Normal"
        else:
            return "Clone 0"
    else:
        lookup = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ]

        roman_numeral = ""

        for value, symbol in lookup:
            while num >= value:
                roman_numeral += symbol
                num -= value

        return f"Clone {roman_numeral}"
