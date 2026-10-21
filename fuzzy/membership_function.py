# membership_functions.py
import numpy as np

def trimf(x: np.ndarray, abc: tuple) -> np.ndarray:
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    # rising edge
    idx = (a < b)
    if idx:
        left = (x >= a) & (x <= b)
        y[left] = (x[left] - a) / (b - a)
    # falling edge
    idx2 = (b < c)
    if idx2:
        right = (x >= b) & (x <= c)
        y[right] = (c - x[right]) / (c - b)
    y[x == b] = 1.0
    return np.clip(y, 0.0, 1.0)

def trapmf(x: np.ndarray, abcd: tuple) -> np.ndarray:
    a, b, c, d = abcd
    y = np.zeros_like(x, dtype=float)
    # rising
    rising = (x >= a) & (x < b)
    if b != a:
        y[rising] = (x[rising] - a) / (b - a)
    # top
    top = (x >= b) & (x <= c)
    y[top] = 1.0
    # falling
    falling = (x > c) & (x <= d)
    if d != c:
        y[falling] = (d - x[falling]) / (d - c)
    return np.clip(y, 0.0, 1.0)
