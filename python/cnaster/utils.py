from copyreg import pickle
import os
import h5py
import pickle
import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from pathlib import Path
from functools import wraps
from numba import njit
from cnaster.config import get_global_config

logger = logging.getLogger(__name__)

def cacher(filename):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_global_config()
            
            output_dir = config.paths.output_dir
            filepath = os.path.join(output_dir, "cache", filename)

            ext = os.path.splitext(filepath)[1].lower()

            # TODO
            def write_h5(d, p):
                with h5py.File(p, 'w') as f:
                    if hasattr(d, '_fields'):
                        f.attrs['fields'] = d._fields
                        d = d._asdict()
                    
                    for k, v in d.items():
                        f.create_dataset(k, data=v)

            def load_h5(p):
                with h5py.File(p, 'r') as f:
                    data = {k: f[k][()] for k in f.keys()}
                    
                    if 'fields' in f.attrs:
                        fields = f.attrs['fields']

                        if isinstance(fields, np.ndarray):
                            fields = [x.decode('utf-8') if isinstance(x, bytes) else x for x in fields]
                        
                        GenericTuple = namedtuple("GenericTuple", fields)
                        return GenericTuple(**data)
                    
                    return data
            
            def synopsis_h5(d):
                if hasattr(d, '_fields'):
                    return f"\tfields={d._fields}"
                return f"\tkeys={list(d.keys())}"

            strategies = {
                '.tsv': (
                    lambda p: pd.read_csv(p, sep='\t'), 
                    lambda d, p: d.to_csv(p, sep='\t', index=False),
                    lambda d: d.head()
                ),
                '.csv': (
                    lambda p: pd.read_csv(p), 
                    lambda d, p: d.to_csv(p, index=False),
                    lambda d: d.head()
                ),
                '.pkl': (
                    lambda p: pickle.load(open(p, "rb")), 
                    lambda d, p: pickle.dump(d, open(p, "wb")),
                    lambda d: f"\ttype={type(d)}"
                ),
                '.npy': (
                    lambda p: np.load(p), 
                    lambda d, p: np.save(p, d),
                    lambda d: f"{d}"
                ),
                '.hdf5': (
                    load_h5, 
                    write_h5,
                    synopsis_h5
                ),
            }

            if ext not in strategies:
                logger.warning(f"Skipping unknown extension '{ext}' for caching:\n'{filepath}'.")
                return func(*args, **kwargs)

            loader, writer, synopsis = strategies.get(ext, strategies['.pkl'])

            if config.run.cache and os.path.exists(filepath):
                mtime = os.path.getmtime(filepath)
                last_modified = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

                result = loader(filepath)

                logger.warning(f"Loading cached result (last modified: {last_modified}) from:\n{filepath}\n with result:\n{synopsis(result)}")

                return result

            result = func(*args, **kwargs)

            if config.run.cache and result is not None:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)

                logger.warning(f"Writing cached result to:\n{filepath}.")
                writer(result, filepath)

            return result

        return wrapper

    return decorator

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

    logger.info(f"Writing figure to:\n{opath}")
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
