"""
03_temple_lag.py
Dataset: Temple's own release  (data.zip)

Real layout (confirmed from 00_inspect):
    temple_heart_rate_validation_data/clean/<participant>/<device>.csv
    each CSV has columns  t_s, hr_bpm   (devices are on different, irregular clocks)
    reference device = polar_h10_taped.csv

For each device, vs the taped Polar H10 reference, it computes:
  * recovery-segment error (MAPE, %) while HR is coming down, and
  * the time-lag (s).
It also prints OVERALL MAPE as a sanity check - Temple's overall MAPE should land
near the 0.98% in their paper. That match proves the pipeline is genuine; the
DECLINE-segment numbers below are new (not in their paper).
Every device is resampled onto a common 1 s grid before comparison. No simulated
data, no hard-coded results.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hrlib as H

TEMPLE_ROOT = "data_extracted/temple"
SUBSET = "clean"                     # use Temple's cleaned per-device HR
REF_FILE = "polar_h10_taped.csv"     # taped Polar H10 = gold-standard reference
GRID_FS = 1.0
GAP_MAX_S = 10.0
DEVICE_FILES = {
    "temple.csv":                 "Temple",
    "apple_watch_s12.csv":        "Apple Watch S12",
    "fitbit_air.csv":             "Fitbit Air",
    "garmin_forerunner_970.csv":  "Garmin FR970",
    "oura_ring_5.csv":            "Oura Ring 5",
    "whoop_mg.csv":               "WHOOP MG",
}


def load_series(path):
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    tc = low.get("t_s", df.columns[0])
    hc = low.get("hr_bpm", df.columns[1])
    t = pd.to_numeric(df[tc], errors="coerce").to_numpy(float)
    hr = pd.to_numeric(df[hc], errors="coerce").to_numpy(float)
    return t, hr


all_refs = glob.glob(os.path.join(TEMPLE_ROOT, "**", REF_FILE), recursive=True)
refs = [p for p in all_refs if (os.sep + SUBSET + os.sep) in p]
if not refs:
    raise SystemExit(f"No {REF_FILE} found under a '{SUBSET}' folder - check 00_inspect.")
print(f"{len(refs)} participant folders found (subset = {SUBSET})")

res = {name: {"lag": [], "fall": [], "all": []} for name in DEVICE_FILES.values()}
example = None
for ref_path in sorted(refs):
    pdir = os.path.dirname(ref_path)
    pid = os.path.basename(pdir)
    t_ref, hr_ref = load_series(ref_path)
    for fname, name in DEVICE_FILES.items():
        dev_path = os.path.join(pdir, fname)
        if not os.path.exists(dev_path):
            continue
        t_dev, hr_dev = load_series(dev_path)
        aligned = H.align_pair(t_ref, hr_ref, t_dev, hr_dev, GRID_FS, GAP_MAX_S)
        if aligned is None:
            continue
        grid, ref, dev, valid = aligned
        lag, _ = H.best_lag_seconds(ref, dev, GRID_FS)   # continuous series
        fall = H.falling_mask(ref, GRID_FS)
        mf, nf = H.mape(ref, dev, valid & fall)          # scored only on real data
        ma, _ = H.mape(ref, dev, valid)
        res[name]["lag"].append(lag)
        res[name]["fall"].append(mf)
        res[name]["all"].append(ma)
        if name == "Temple" and example is None and nf > 20:
            example = (pid, grid, ref, dev)

print("\n==== TEMPLE DATASET SUMMARY (median over participants) ====")
print(f"{'device':16s} {'decline-MAPE%':>14s} {'overall-MAPE%':>14s} {'lag(s)':>9s} {'n':>5s}")
summary = {}
for name in DEVICE_FILES.values():
    if not res[name]["fall"]:
        continue
    mf = np.nanmedian(res[name]["fall"])
    ma = np.nanmedian(res[name]["all"])
    lg = np.nanmedian(res[name]["lag"])
    summary[name] = (mf, ma, lg)
    print(f"{name:16s} {mf:14.3f} {ma:14.3f} {lg:+9.2f} {len(res[name]['fall']):5d}")

print("\n  sanity : Temple overall-MAPE should be close to the 0.98% in the paper.")
print("  headline: the DECLINE-MAPE column is your own number (not in the paper).")
print("  sentence: Temple decline-MAPE  vs  the wrist/finger devices' decline-MAPE.")

names = list(summary.keys())
mf_vals = [summary[n][0] for n in names]
order = np.argsort(mf_vals)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
if example:
    pid, g, ref, dev = example
    ax[0].plot(g, ref, lw=2, label="Polar H10 reference")
    ax[0].plot(g, dev, lw=1.5, alpha=.85, label="Temple HR")
    ax[0].set_title(f"Temple dataset: participant {pid}")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("HR (bpm)"); ax[0].legend()
ax[1].bar([names[i] for i in order], [mf_vals[i] for i in order],
          color=["#2a9d8f" if names[i] == "Temple" else "#9aa0a6" for i in order])
ax[1].set_ylabel("decline-MAPE (%)")
ax[1].set_title("Recovery error by device (lower = better)")
plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("fig_temple_lag.png", dpi=150)
print("saved fig_temple_lag.png")
