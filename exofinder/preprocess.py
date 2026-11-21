"""Label-blind preprocessing. Time is always BKJD; duration is in hours."""
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import binned_statistic

GLOBAL_BINS = 512
LOCAL_BINS = 128


def phase_days(time, period, epoch):
    if not np.isfinite([period, epoch]).all() or period <= 0:
        raise ValueError("Period must be positive and ephemeris finite.")
    return (np.asarray(time) - epoch + period / 2) % period - period / 2


def bin_view(phase, flux, low, high, bins):
    edges = np.linspace(low, high, bins + 1)
    values = binned_statistic(phase, flux, statistic="median", bins=edges)[0]
    mask = np.isfinite(values)
    if mask.sum() < 8:
        raise ValueError("Insufficient phase coverage: fewer than eight populated bins.")
    # Missing bins are neutral, with an explicit occupancy channel; never invent a dip.
    return np.stack([np.where(mask, values, 0), mask.astype(float)]).astype("float32")


def prepare(time, flux, period, epoch, duration):
    if not np.isfinite(duration) or duration <= 0 or duration / 24 >= period / 4:
        raise ValueError("Transit duration must be positive and shorter than a quarter-period.")
    t, f = np.asarray(time, float), np.asarray(flux, float)
    valid = np.isfinite(t) & np.isfinite(f) & (f > 0)
    t, f = t[valid], f[valid]
    order = np.argsort(t)
    t, f = t[order], f[order]
    t, unique = np.unique(t, return_index=True)
    f = f[unique]
    if len(t) < 200 or np.ptp(t) < 2 * period:
        raise ValueError("Need 200 valid observations spanning at least two periods.")
    f = f / np.median(f)
    p = phase_days(t, period, epoch)
    outside = np.abs(p) > duration / 24 * 1.5
    if outside.sum() < 100:
        raise ValueError("Insufficient out-of-transit observations.")
    # Mask primary transit before detrending, then handle long cadence gaps independently.
    filled = np.interp(t, t[outside], f[outside])
    cadence = np.median(np.diff(t))
    window = max(51, int(max(2, 5 * duration / 24) / cadence) | 1)
    trend = np.empty_like(f)
    segments = np.split(np.arange(len(t)), np.where(np.diff(t) > 0.5)[0] + 1)
    for seg in segments:
        trend[seg] = median_filter(filled[seg], size=min(window, (len(seg) // 2) * 2 + 1), mode="nearest")
    flat = f / np.maximum(trend, 1e-8) - 1
    center = np.median(flat[outside])
    noise = max(1.4826 * np.median(np.abs(flat[outside] - center)), 1e-6)
    # A per-object robust scale retains depth in S/N units without catalog depth leakage.
    normalized = np.clip((flat - center) / noise, -50, 50)
    g = bin_view(p, normalized, -period / 2, period / 2, GLOBAL_BINS)
    extent = min(2.5 * duration / 24, period / 2)
    local = bin_view(p, normalized, -extent, extent, LOCAL_BINS)
    if local[1].mean() < 0.25:
        raise ValueError("Less than 25% local phase-bin coverage.")
    return g, local, {"noise_ppm": float(noise * 1e6), "cadences": len(t), "local_extent_days": extent}


def summary_features(global_view, local_view):
    """Shape-only baseline; uses exactly the CNN's input, not catalog labels/flags."""
    g, l = global_view[:, 0], local_view[:, 0]
    return np.column_stack([np.min(g, axis=1), np.std(g, axis=1), np.min(l, axis=1),
                            np.std(l, axis=1), np.mean(l[:, 48:80], axis=1),
                            np.mean(g[:, :32], axis=1), np.mean(g[:, -32:], axis=1)])
