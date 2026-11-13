"""
02_uci_lag.py
Dataset: UCI PPG-DaLiA  (ppg+dalia.zip)

For each subject it computes, from the real signals:
  * the time-lag (s) between wrist-PPG HR and the chest-ECG ground-truth HR, and
  * the recovery-segment error (MAPE, %) while HR is coming down.
The wrist-PPG HR is computed here from the Empatica BVP on the SAME 8 s / 2 s
windows the dataset uses for its ECG ground truth, so the comparison is fair.
"""
import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import hrlib as H

UCI_DIR = "data_extracted/uci"
FS_BVP = 64.0                 # Empatica E4 BVP sampling rate (dataset spec)
GRID_FS = 1.0 / H.STEP_S      # ground-truth HR is one value every 2 s

pkls = sorted(glob.glob(os.path.join(UCI_DIR, "**", "*.pkl"), recursive=True))
print(f"found {len(pkls)} subject files")
if not pkls:
    raise SystemExit("No .pkl files - check 00_inspect output / UCI_DIR.")

rows, example = [], None
for p in pkls:
    with open(p, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    bvp = np.asarray(d["signal"]["wrist"]["BVP"], float).ravel()
    acc = np.asarray(d["signal"]["wrist"]["ACC"], float)   # (N,3) @ 32 Hz
    ref = np.asarray(d["label"], float).ravel()        # ECG-derived HR, 0.5 Hz

    # motion-compensated wrist HR, same 8 s / 2 s windows as the ground truth
    t_dev, hr_dev = H.ppg_to_hr_motioncomp(bvp, FS_BVP, acc, 32.0)
    # ground-truth window k spans [2k, 2k+8] s -> centre 2k+4 s (same as t_dev)
    n = min(len(hr_dev), len(ref))
    if n < 10:
        print(f"  skip {os.path.basename(p)}: too short")
        continue
    hr_dev, hr_ref = hr_dev[:n], ref[:n]
    t_ref = 2.0 * np.arange(n) + H.WIN_S / 2.0

    lag, _ = H.best_lag_seconds(hr_ref, hr_dev, GRID_FS)
    fall = H.falling_mask(hr_ref, GRID_FS)
    mape_fall, n_fall = H.mape(hr_ref, hr_dev, fall)
    mape_all, _ = H.mape(hr_ref, hr_dev)
    sid = os.path.basename(p).replace(".pkl", "")
    rows.append((sid, lag, mape_fall, mape_all, n_fall))
    print(f"  {sid:6s} lag={lag:+.2f}s  MAPE_decline={mape_fall:7.3f}%  "
          f"MAPE_all={mape_all:7.3f}%  (n_decline={n_fall})")
    if example is None and n_fall > 30:
        example = (sid, t_ref, hr_ref, hr_dev)

if not rows:
    raise SystemExit("No usable subjects.")

lags = np.array([r[1] for r in rows], float)
mfall = np.array([r[2] for r in rows], float)
med_lag, med_mape = np.nanmedian(lags), np.nanmedian(mfall)
print("\n==== UCI PPG-DaLiA SUMMARY (median over subjects) ====")
print(f"  median decline-lag          : {med_lag:+.3f} s")
print(f"  median decline-MAPE (error) : {med_mape:.3f} %   <-- UCI number")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
if example:
    sid, t, ref_, dev_ = example
    ax[0].plot(t, ref_, lw=2, label="ECG ground-truth HR")
    ax[0].plot(t, dev_, lw=1.5, alpha=.85, label="wrist-PPG HR")
    ax[0].set_title(f"UCI PPG-DaLiA: {sid}")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("HR (bpm)"); ax[0].legend()
ax[1].bar(range(len(rows)), mfall, color="#9aa0a6")
ax[1].set_xticks(range(len(rows)))
ax[1].set_xticklabels([r[0] for r in rows], rotation=90, fontsize=7)
ax[1].axhline(med_mape, ls="--", color="k", label=f"median {med_mape:.2f}%")
ax[1].set_ylabel("decline-MAPE (%)")
ax[1].set_title("Wrist-PPG recovery error per subject")
ax[1].legend()
plt.tight_layout()
plt.savefig("fig_uci.png", dpi=150)
print("saved fig_uci.png")
