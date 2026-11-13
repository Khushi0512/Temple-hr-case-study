"""
hrlib.py - shared, deterministic helpers for HR-from-signal and tracking metrics.

Design rules:
  * No random numbers anywhere. Nothing is simulated or injected.
  * No hard-coded result values. Every number is computed from the data.
  * ONE single method is used across all three datasets, so the results are
    comparable and the pipeline is internally consistent.
  * Reference HR always comes from an independent ECG source; device HR always
    comes from the optical/PPG source. We never compare a signal against itself.

All parameters below are physiological constants, documented inline.
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend

# ---- physiological constants (fixed, not tuned) ----
HR_MIN_BPM = 30.0                                   # lowest plausible human HR
HR_MAX_BPM = 240.0                                  # highest plausible human HR
PPG_BAND_HZ = (HR_MIN_BPM / 60.0, HR_MAX_BPM / 60.0)  # 0.5 - 4.0 Hz
WIN_S = 8.0           # window for one HR estimate (standard for PPG HR)
STEP_S = 2.0          # hop between windows -> one HR value every 2 s
MAX_LAG_S = 30.0      # search range for device-vs-reference time offset
SLOPE_SMOOTH_S = 15.0 # smoothing applied to reference HR before taking its slope


def bandpass(x, fs, band=PPG_BAND_HZ, order=4):
    ny = 0.5 * fs
    b, a = butter(order, [band[0] / ny, band[1] / ny], btype="band")
    return filtfilt(b, a, x)


def ppg_to_hr(ppg, fs, win_s=WIN_S, step_s=STEP_S):
    """Windowed dominant-frequency HR from a PPG/BVP trace.
    Standard baseline: band-pass to the human-HR band, then take the strongest
    spectral peak in each window. Returns (t_centres_sec, hr_bpm)."""
    ppg = np.asarray(ppg, float).ravel()
    if ppg.size == 0:
        return np.array([]), np.array([])
    # some recordings have NaN/inf gaps (e.g. PhysioNet wrist PPG). Interpolate
    # across them so filtering and the FFT stay intact instead of crashing.
    idx = np.arange(ppg.size)
    good = np.isfinite(ppg)
    if good.sum() < 8:
        return np.array([]), np.array([])
    if not good.all():
        ppg = np.interp(idx, idx[good], ppg[good])
    ppg = detrend(ppg)
    ppg = bandpass(ppg, fs)
    win = int(round(win_s * fs))
    step = int(round(step_s * fs))
    if win < 8 or len(ppg) < win:
        return np.array([]), np.array([])
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band = (freqs >= PPG_BAND_HZ[0]) & (freqs <= PPG_BAND_HZ[1])
    fb = freqs[band]
    w = np.hanning(win)
    t, hr = [], []
    for s in range(0, len(ppg) - win + 1, step):
        spec = np.abs(np.fft.rfft(ppg[s:s + win] * w))[band]
        hr.append(fb[int(np.argmax(spec))] * 60.0)
        t.append((s + win / 2.0) / fs)          # window centre, seconds
    return np.asarray(t), np.asarray(hr)


def instantaneous_hr(rpeak_samples, fs):
    """From ECG R-peak sample indices -> (beat_times_sec, inst_HR_bpm), filtered
    to physiologically possible values."""
    r = np.asarray(rpeak_samples, float) / fs
    if len(r) < 3:
        return np.array([]), np.array([])
    rr = np.diff(r)
    hr = 60.0 / rr
    tb = r[1:]
    ok = (hr >= HR_MIN_BPM) & (hr <= HR_MAX_BPM) & np.isfinite(hr)
    return tb[ok], hr[ok]


def windowed_ref_hr(rpeak_samples, fs, centres, win_s=WIN_S):
    """Reference HR averaged over the SAME 8 s windows as the PPG estimate, so any
    measured lag is physiological rather than a windowing artefact. Each centre's
    value = mean of the instantaneous HR whose beats fall inside that window."""
    tb, hr = instantaneous_hr(rpeak_samples, fs)
    out = np.full(len(centres), np.nan)
    if len(hr) < 2:
        return out
    half = win_s / 2.0
    for i, c in enumerate(centres):
        sel = (tb >= c - half) & (tb < c + half)
        if sel.sum() >= 2:
            out[i] = hr[sel].mean()
    return out


def best_lag_seconds(ref_hr, dev_hr, fs_grid, max_lag_s=MAX_LAG_S):
    """Integer-sample cross-correlation lag that best aligns dev to ref.
    Positive lag => the device TRAILS the reference (reacts late). Deterministic."""
    ref = np.asarray(ref_hr, float)
    dev = np.asarray(dev_hr, float)
    m = np.isfinite(ref) & np.isfinite(dev)
    ref, dev = ref[m], dev[m]
    if len(ref) < 10:
        return np.nan, np.nan
    ref = (ref - ref.mean()) / (ref.std() + 1e-12)
    dev = (dev - dev.mean()) / (dev.std() + 1e-12)
    max_lag = int(round(max_lag_s * fs_grid))
    best_r, best_lag, n = -np.inf, 0, len(ref)
    for L in range(-max_lag, max_lag + 1):
        if L >= 0:
            a, b = dev[L:], ref[:n - L]
        else:
            a, b = dev[:n + L], ref[-L:]
        if len(a) < 10:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r, best_lag = r, L
    return best_lag / fs_grid, best_r


def falling_mask(ref_hr, fs_grid, smooth_s=SLOPE_SMOOTH_S):
    """True where the smoothed reference HR is decreasing (recovery / decline).
    Only the sign of the smoothed slope is used - no magic threshold."""
    ref = np.asarray(ref_hr, float)
    k = max(1, int(round(smooth_s * fs_grid)))
    kern = np.ones(k) / k
    filled = np.nan_to_num(ref, nan=np.nanmean(ref))
    sm = np.convolve(filled, kern, mode="same")
    return np.gradient(sm) < 0


def mape(ref, dev, mask=None):
    """Mean absolute percentage error (%) and the number of points used."""
    ref = np.asarray(ref, float)
    dev = np.asarray(dev, float)
    m = np.isfinite(ref) & np.isfinite(dev) & (ref > 0)
    if mask is not None:
        m = m & np.asarray(mask, bool)
    if m.sum() == 0:
        return np.nan, 0
    return float(np.mean(np.abs(dev[m] - ref[m]) / ref[m]) * 100.0), int(m.sum())


def align_pair(t_ref, hr_ref, t_dev, hr_dev, grid_fs=1.0, gap_max_s=10.0):
    """Put a reference series and a device series onto ONE uniform time grid.
    Needed because Temple's devices report on different, irregular clocks
    (e.g. Apple Watch ~every 5 s, Fitbit every 1 s). Each series is interpolated
    from its own timestamps after NaNs are dropped; any grid point farther than
    gap_max_s from a real sample of a series is flagged invalid for that series,
    so genuine dropouts are never invented. Returns (grid, ref, dev, valid_mask)
    or None if there isn't enough overlap."""
    t_ref = np.asarray(t_ref, float); hr_ref = np.asarray(hr_ref, float)
    t_dev = np.asarray(t_dev, float); hr_dev = np.asarray(hr_dev, float)

    def clean(t, x):
        m = np.isfinite(t) & np.isfinite(x)
        order = np.argsort(t[m])
        return t[m][order], x[m][order]

    tr, xr = clean(t_ref, hr_ref)
    td, xd = clean(t_dev, hr_dev)
    if len(tr) < 5 or len(td) < 5:
        return None
    t0 = max(tr.min(), td.min())
    t1 = min(tr.max(), td.max())
    if t1 - t0 < 10:
        return None
    grid = np.arange(np.ceil(t0), np.floor(t1) + 1, 1.0 / grid_fs)

    def interp_with_mask(t, x):
        y = np.interp(grid, t, x)                 # fills everything (continuous)
        idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
        near = np.minimum(np.abs(grid - t[idx - 1]), np.abs(t[idx] - grid))
        return y, near <= gap_max_s               # True where a real sample is close
    yr, mr = interp_with_mask(tr, xr)
    yd, md = interp_with_mask(td, xd)
    return grid, yr, yd, (mr & md)


def ppg_to_hr_motioncomp(ppg, fs, acc, acc_fs=None, win_s=WIN_S, step_s=STEP_S):
    """Motion-compensated windowed HR from PPG, using the accelerometer as a
    motion reference (so cadence during walking/running stops being mistaken for
    heart rate). For each window:
      1. take the PPG and the summed-axis ACC spectra on the same grid,
      2. if real motion is present (ACC peak well above its own baseline),
         null the PPG bins at the ACC's dominant frequency and its 2nd harmonic
         (within ~1.5 FFT bins), and
      3. take the strongest remaining PPG peak in the human-HR band.
    A final median filter removes isolated spikes. Nothing here touches the
    ECG / ground truth - only the device's own PPG and ACC. All constants are
    fixed physiological choices, not tuned to any result.
    """
    from scipy.signal import medfilt
    ppg = np.asarray(ppg, float).ravel()
    n = ppg.size
    if n == 0:
        return np.array([]), np.array([])
    gi = np.arange(n); good = np.isfinite(ppg)
    if good.sum() < 8:
        return np.array([]), np.array([])
    if not good.all():
        ppg = np.interp(gi, gi[good], ppg[good])
    ppg = bandpass(detrend(ppg), fs)

    # accelerometer axes, interpolated over NaN and resampled onto the PPG grid
    acc = np.atleast_2d(np.asarray(acc, float))
    if acc.shape[0] > acc.shape[1]:
        acc = acc.T                                   # -> (axes, samples)
    axes = []
    for a in acc:
        a = np.asarray(a, float)
        g = np.isfinite(a)
        if g.sum() < 8:
            continue
        if not g.all():
            a = np.interp(np.arange(a.size), np.arange(a.size)[g], a[g])
        if acc_fs and abs(acc_fs - fs) > 1e-6:
            a = np.interp(np.arange(n) / fs, np.arange(a.size) / acc_fs, a)
        axes.append(bandpass(detrend(a), fs))

    win = int(round(win_s * fs)); step = int(round(step_s * fs))
    if win < 8 or n < win:
        return np.array([]), np.array([])
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    band = (freqs >= PPG_BAND_HZ[0]) & (freqs <= PPG_BAND_HZ[1])
    fb = freqs[band]
    tol = 1.5 * (1.0 / win_s)                          # ~1.5 FFT bins
    w = np.hanning(win)
    t, hr = [], []
    for s in range(0, n - win + 1, step):
        P = np.abs(np.fft.rfft(ppg[s:s + win] * w))[band]
        A = np.zeros_like(P)
        for a in axes:
            if s + win <= a.size:
                A = A + np.abs(np.fft.rfft(a[s:s + win] * w))[band]
        Pc = P.copy()
        if A.size and A.max() > 3.0 * (np.median(A) + 1e-12):   # motion present
            fa = fb[int(np.argmax(A))]
            for harm in (fa, 2.0 * fa):
                Pc[np.abs(fb - harm) <= tol] = 0.0
        if not np.any(Pc > 0):
            Pc = P
        hr.append(fb[int(np.argmax(Pc))] * 60.0)
        t.append((s + win / 2.0) / fs)
    hr = np.asarray(hr); t = np.asarray(t)
    if hr.size >= 5:
        hr = medfilt(hr, kernel_size=5)
    return t, hr
