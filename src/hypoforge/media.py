"""Explicit, deterministic MMM-inspired feature transforms (not fitted MMM models)."""
import numpy as np
import pandas as pd


def _numeric(x):
    s = pd.Series(x, dtype='float64')
    if not np.isfinite(s.dropna()).all():
        raise ValueError('Input must contain finite values or NA')
    return s


def geometric_adstock(x, alpha=0.5, l_max=8, *, normalize=True):
    """Finite geometric weighted lag, with no future leakage.

    NA inputs are treated as zero exposure; leading lags use available history.
    """
    if not 0 <= alpha < 1 or not isinstance(l_max, int) or l_max < 0:
        raise ValueError('Require 0 <= alpha < 1 and integer l_max >= 0')
    s = _numeric(x).fillna(0)
    weights = alpha ** np.arange(l_max + 1)
    values = s.to_numpy()
    out = np.zeros(len(s))
    for t in range(len(s)):
        w = weights[:min(t, l_max) + 1]
        out[t] = np.dot(w, values[t - len(w) + 1:t + 1][::-1]) / (w.sum() if normalize else 1)
    return pd.Series(out, index=s.index, name=s.name)


def delayed_adstock(x, alpha=0.5, peak=2, l_max=12, *, normalize=True):
    """Finite lag kernel alpha ** abs(lag - peak); peak delay is integer."""
    if not 0 < alpha < 1 or not isinstance(peak, int) or peak < 0 or not isinstance(l_max, int) or l_max < peak:
        raise ValueError('Require 0 < alpha < 1 and 0 <= peak <= l_max')
    s = _numeric(x).fillna(0)
    w_all = alpha ** np.abs(np.arange(l_max + 1) - peak)
    out = np.zeros(len(s))
    a = s.to_numpy()
    for t in range(len(s)):
        w = w_all[:min(t, l_max) + 1]
        out[t] = np.dot(w, a[t-len(w)+1:t+1][::-1]) / (w.sum() if normalize else 1)
    return pd.Series(out, index=s.index)


def weibull_adstock(x, shape=2.0, scale=3.0, l_max=12, *, normalize=True):
    """Discrete Weibull survival-difference kernel, indexed at lag zero."""
    if shape <= 0 or scale <= 0 or not isinstance(l_max, int) or l_max < 0:
        raise ValueError('Require positive shape/scale and nonnegative integer l_max')
    s = _numeric(x).fillna(0)
    lag = np.arange(l_max + 1, dtype=float)
    w_all = np.exp(-((lag / scale) ** shape)) - np.exp(-(((lag + 1) / scale) ** shape))
    a = s.to_numpy(); out = np.zeros(len(s))
    for t in range(len(s)):
        w = w_all[:min(t, l_max) + 1]
        out[t] = np.dot(w, a[t-len(w)+1:t+1][::-1]) / (w.sum() if normalize else 1)
    return pd.Series(out, index=s.index)


def logistic_saturation(x, lam=1.0):
    """Zero-based logistic response 2/(1+exp(-lam*x))-1, for x >= 0."""
    if lam <= 0: raise ValueError('lam must be > 0')
    s = _numeric(x)
    if (s.dropna() < 0).any(): raise ValueError('Saturation requires nonnegative x')
    return pd.Series(np.tanh(lam * s.to_numpy() / 2), index=s.index)


def michaelis_menten(x, half_saturation=1.0, maximum=1.0):
    if half_saturation <= 0 or maximum <= 0: raise ValueError('Parameters must be > 0')
    s = _numeric(x)
    if (s.dropna() < 0).any(): raise ValueError('Saturation requires nonnegative x')
    return maximum * s / (half_saturation + s)


def power_saturation(x, exponent=0.5):
    if not 0 < exponent <= 1: raise ValueError('Require 0 < exponent <= 1')
    s = _numeric(x)
    if (s.dropna() < 0).any(): raise ValueError('Saturation requires nonnegative x')
    return s.pow(exponent)


def tanh_saturation(x, scale=1.0):
    if scale <= 0: raise ValueError('scale must be > 0')
    s = _numeric(x)
    if (s.dropna() < 0).any(): raise ValueError('Saturation requires nonnegative x')
    return pd.Series(np.tanh(s.to_numpy() / scale), index=s.index)


def reach_frequency(reach, frequency, *, half_saturation=2.0, slope=1.0):
    """Reach times Hill-saturated average frequency; no adstock applied."""
    if half_saturation <= 0 or slope <= 0: raise ValueError('Parameters must be > 0')
    r, f = _numeric(reach), _numeric(frequency)
    if len(r) != len(f) or (r.dropna() < 0).any() or (f.dropna() < 0).any():
        raise ValueError('Matching nonnegative reach and frequency required')
    a = f.pow(slope)
    return r * a / (half_saturation ** slope + a)
