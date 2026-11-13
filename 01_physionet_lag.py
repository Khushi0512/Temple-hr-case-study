"""
01_physionet_lag.py
Dataset: PhysioNet 'Wrist PPG During Exercise'  (wrist-1.0.0.zip)

For each record it computes, from the real signals:
  * the time-lag (s) between wrist-PPG HR and the chest-ECG reference HR, and
  * the recovery-segment error (MAPE, %) while HR is coming down.
Reference HR is built from the ECG on the SAME 8 s windows as the PPG estimate,
so the measured lag is physiological, not a windowing artefact.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from wfdb import processing as wp
import hrlib as H

PHYS_DIR = "data_extracted/physionet"   # produced by 00_inspect.py
GRID_FS = 1.0 / H.STEP_S                 # HR series rate = one value every 2 s


def record_dir(root):
    hits = glob.glob(os.path.join(root, "**", "RECORDS"), recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    heas = glob.glob(os.path.join(root, "**", "*.hea"), recursive=True)
    if heas:
        return os.path.dirname(heas[0])
    raise FileNotFoundError("No WFDB records found under " + root)


def annotator_ext(rec_dir):
    f = os.path.join(rec_dir, "ANNOTATORS")
    if os.path.exists(f):
        with open(f) as fh:
            line = fh.readline().strip()
        if line:
            return line.replace("\t", " ").split()[0]
    return None


rec_dir = record_dir(PHYS_DIR)
ann_ext = annotator_ext(rec_dir)
with open(os.path.join(rec_dir, "RECORDS")) as fh:
    records = [l.strip() for l in fh if l.strip()]
print(f"{len(records)} records in {rec_dir} | annotator = {ann_ext}")

rows, example = [], None
for rec in records:
    path = os.path.join(rec_dir, rec)
    try:
        r = wfdb.rdrecord(path)
    except Exception as e:
        print(f"  skip {rec}: {e}")
        continue
    names = [s.lower() for s in r.sig_name]
    ppg_idx = next((i for i, n in enumerate(names) if "ppg" in n), None)
    ecg_idx = next((i for i, n in enumerate(names) if "ecg" in n), None)
    if ppg_idx is None or ecg_idx is None:
        print(f"  skip {rec}: channels = {r.sig_name}")
        continue
    fs = r.fs
    ppg = r.p_signal[:, ppg_idx]
    acc_idx = [i for i, nm in enumerate(names) if "accel" in nm]
    acc = r.p_signal[:, acc_idx].T if acc_idx else None

    # reference R-peaks: prefer the provided annotation, else detect from ECG
    rpk, fs_ann = None, fs
    if ann_ext:
        try:
            ann = wfdb.rdann(path, ann_ext)
            rpk = ann.sample
            fs_ann = ann.fs or fs
        except Exception:
            rpk = None
    if rpk is None:
        ecg = r.p_signal[:, ecg_idx]
        det = wp.XQRS(sig=ecg, fs=fs)
        det.detect(verbose=False)
        rpk = np.asarray(det.qrs_inds)
        fs_ann = fs
    rpk = np.asarray(rpk, float) * (fs / fs_ann)      # -> ppg sample space

    if acc is not None and acc.shape[0] >= 1:         # motion-compensated HR
        t_dev, hr_dev = H.ppg_to_hr_motioncomp(ppg, fs, acc, fs)
    else:
        t_dev, hr_dev = H.ppg_to_hr(ppg, fs)
    if len(t_dev) < 10:
        print(f"  skip {rec}: too short")
        continue
    hr_ref = H.windowed_ref_hr(rpk, fs, t_dev)        # ECG HR, matched windows

    lag, _ = H.best_lag_seconds(hr_ref, hr_dev, GRID_FS)
    fall = H.falling_mask(hr_ref, GRID_FS)
    mape_fall, n_fall = H.mape(hr_ref, hr_dev, fall)
    mape_all, _ = H.mape(hr_ref, hr_dev)
    rows.append((rec, lag, mape_fall, mape_all, n_fall))
    print(f"  {rec:28s} lag={lag:+.2f}s  MAPE_decline={mape_fall:7.3f}%  "
          f"MAPE_all={mape_all:7.3f}%  (n_decline={n_fall})")
    if example is None and n_fall > 20:
        example = (rec, t_dev, hr_ref, hr_dev)

if not rows:
    raise SystemExit("No usable records - check 00_inspect output.")

lags = np.array([r[1] for r in rows], float)
mfall = np.array([r[2] for r in rows], float)
med_lag, med_mape = np.nanmedian(lags), np.nanmedian(mfall)
print("\n==== PHYSIONET SUMMARY (median over records) ====")
print(f"  median decline-lag          : {med_lag:+.3f} s")
print(f"  median decline-MAPE (error) : {med_mape:.3f} %   <-- PhysioNet number")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
if example:
    rec, t, ref, dev = example
    ax[0].plot(t, ref, lw=2, label="ECG reference HR")
    ax[0].plot(t, dev, lw=1.5, alpha=.85, label="wrist-PPG HR")
    ax[0].set_title(f"PhysioNet: {rec}")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("HR (bpm)"); ax[0].legend()
ax[1].bar(range(len(rows)), mfall, color="#9aa0a6")
ax[1].set_xticks(range(len(rows)))
ax[1].set_xticklabels([r[0] for r in rows], rotation=90, fontsize=7)
ax[1].axhline(med_mape, ls="--", color="k", label=f"median {med_mape:.2f}%")
ax[1].set_ylabel("decline-MAPE (%)")
ax[1].set_title("Wrist-PPG recovery error per record")
ax[1].legend()
plt.tight_layout()
plt.savefig("fig_physionet.png", dpi=150)
print("saved fig_physionet.png")
