"""
04_temple_zone_error.py
Dataset: Temple's own release  (data.zip)

NEW metric, NOT in their paper: for each device, the % of seconds it places the
athlete in a DIFFERENT heart-rate training zone than the Polar H10 reference does.

Layout (from 00_inspect):
    temple_heart_rate_validation_data/participants.csv   (has age, gender)
    temple_heart_rate_validation_data/clean/<participant>/<device>.csv  (t_s, hr_bpm)

Zones are % of HRmax (bands 50/60/70/80/90). HRmax per participant uses the
Tanaka formula the paper cites: HRmax = 208 - 0.7 * age, read from participants.csv.
If a participant's age can't be matched, that participant falls back to their peak
reference HR as an HRmax proxy.
Devices are resampled onto a common 1 s grid before zones are compared.
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hrlib as H

TEMPLE_ROOT = "data_extracted/temple"
SUBSET = "clean"
REF_FILE = "polar_h10_taped.csv"
GRID_FS = 1.0
GAP_MAX_S = 10.0
ZONE_EDGES = [0.50, 0.60, 0.70, 0.80, 0.90]   # fractions of HRmax -> zones 0..5
DEVICE_FILES = {
    "temple.csv":                 "Temple",
    "apple_watch_s12.csv":        "Apple Watch S12",
    "fitbit_air.csv":             "Fitbit Air",
    "garmin_forerunner_970.csv":  "Garmin FR970",
    "oura_ring_5.csv":            "Oura Ring 5",
    "whoop_mg.csv":               "WHOOP MG",
}


def digits(s):
    m = re.findall(r"\d+", str(s))
    return int(m[0]) if m else None


def parse_age(s):
    """Age is published as a band (e.g. '20-24') for privacy. Use the band
    midpoint. HRmax varies ~0.7 bpm/yr, so a 5-yr band shifts HRmax <2 bpm -
    negligible against ~19 bpm-wide zone bands."""
    nums = re.findall(r"\d+", str(s))
    if len(nums) >= 2:
        return (int(nums[0]) + int(nums[1])) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def load_series(path):
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    tc = low.get("t_s", df.columns[0])
    hc = low.get("hr_bpm", df.columns[1])
    t = pd.to_numeric(df[tc], errors="coerce").to_numpy(float)
    hr = pd.to_numeric(df[hc], errors="coerce").to_numpy(float)
    return t, hr


def to_zone(hr, hrmax):
    frac = np.asarray(hr, float) / hrmax
    z = np.zeros_like(frac)
    for i, edge in enumerate(ZONE_EDGES, start=1):
        z[frac >= edge] = i
    z[~np.isfinite(frac)] = np.nan
    return z


# ---- build participant -> age map from participants.csv ----
age_by_pid = {}
pfiles = glob.glob(os.path.join(TEMPLE_ROOT, "**", "participants*.csv"), recursive=True)
if pfiles:
    pdf = pd.read_csv(pfiles[0])
    print(f"participants file: {os.path.relpath(pfiles[0], TEMPLE_ROOT)}  columns={list(pdf.columns)}")
    age_col = next((c for c in pdf.columns if "age" in c.lower()), None)
    # id column = the one whose values look most like the folder ids (digits)
    id_col = None
    for c in pdf.columns:
        if c == age_col:
            continue
        if pdf[c].map(digits).notna().mean() > 0.8:
            id_col = c
            break
    if age_col and id_col:
        for _, row in pdf.iterrows():
            pid = digits(row[id_col])
            age = parse_age(row[age_col])          # band midpoint -> age in years
            if pid is not None and np.isfinite(age):
                age_by_pid[pid] = float(age)
        print(f"  mapped age for {len(age_by_pid)} participants "
              f"(id col='{id_col}', age col='{age_col}', age = band midpoint)")
    else:
        print("  !! couldn't identify id/age columns - will use peak-HR proxy")
else:
    print("no participants.csv found - will use peak-HR proxy for every participant")

all_refs = glob.glob(os.path.join(TEMPLE_ROOT, "**", REF_FILE), recursive=True)
refs = [p for p in all_refs if (os.sep + SUBSET + os.sep) in p]
if not refs:
    raise SystemExit(f"No {REF_FILE} under a '{SUBSET}' folder - check 00_inspect.")
print(f"{len(refs)} participant folders found\n")

wrong_all = {n: 0 for n in DEVICE_FILES.values()}
total_all = {n: 0 for n in DEVICE_FILES.values()}
wrong_hi = {n: 0 for n in DEVICE_FILES.values()}
total_hi = {n: 0 for n in DEVICE_FILES.values()}
n_formula, n_proxy = 0, 0

for ref_path in sorted(refs):
    pdir = os.path.dirname(ref_path)
    pid = digits(os.path.basename(pdir))
    t_ref, hr_ref = load_series(ref_path)
    finite_ref = hr_ref[np.isfinite(hr_ref)]
    if finite_ref.size < 30:
        continue
    if pid in age_by_pid:
        hrmax = 208.0 - 0.7 * age_by_pid[pid]      # Tanaka, as the paper cites
        n_formula += 1
    else:
        hrmax = np.nanmax(finite_ref)              # fallback proxy
        n_proxy += 1
    if not np.isfinite(hrmax) or hrmax < 100:
        continue
    for fname, name in DEVICE_FILES.items():
        dev_path = os.path.join(pdir, fname)
        if not os.path.exists(dev_path):
            continue
        t_dev, hr_dev = load_series(dev_path)
        aligned = H.align_pair(t_ref, hr_ref, t_dev, hr_dev, GRID_FS, GAP_MAX_S)
        if aligned is None:
            continue
        _, ref, dev, valid = aligned
        zr = to_zone(ref, hrmax)
        zd = to_zone(dev, hrmax)
        m = valid & np.isfinite(zr) & np.isfinite(zd)
        wrong_all[name] += int((zr[m] != zd[m]).sum())
        total_all[name] += int(m.sum())
        hi = m & (zr >= 4)
        wrong_hi[name] += int((zr[hi] != zd[hi]).sum())
        total_hi[name] += int(hi.sum())

print(f"HRmax source: {n_formula} participants via age formula, {n_proxy} via peak-HR proxy")

mis_all, mis_hi = {}, {}
for n in DEVICE_FILES.values():
    mis_all[n] = 100.0 * wrong_all[n] / total_all[n] if total_all[n] else np.nan
    mis_hi[n] = 100.0 * wrong_hi[n] / total_hi[n] if total_hi[n] else np.nan

print("\n==== ZONE-MISCLASSIFICATION (% of seconds in the WRONG zone) ====")
print(f"{'device':16s} {'all zones %':>12s} {'zones 4-5 %':>12s}")
for n in DEVICE_FILES.values():
    if total_all[n]:
        print(f"{n:16s} {mis_all[n]:12.3f} {mis_hi[n]:12.3f}")

names = sorted([n for n in mis_all if total_all[n]], key=lambda n: mis_all[n])
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(names, [mis_all[n] for n in names],
       color=["#2a9d8f" if n == "Temple" else "#9aa0a6" for n in names])
ax.set_ylabel("% of seconds in the WRONG training zone")
ax.set_title("Zone-misclassification vs Polar H10 reference (lower = better)")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()
plt.savefig("fig_temple_zone.png", dpi=150)
print("saved fig_temple_zone.png")
