"""
00_inspect.py  -  RUN THIS FIRST.

Unzips (if needed) and prints the structure of all three datasets so the exact
file / column names can be confirmed before anything is computed. Nothing is
analysed here. If the Temple column names differ from what 03/04 expect, paste
this output back and the loaders can be corrected (instead of guessing).
"""
import os
import glob
import zipfile
import pickle
import pandas as pd

# ---- EDIT THESE to point at the zip files you downloaded ----
PHYSIONET_ZIP = "wrist-1.0.0.zip"
UCI_ZIP       = "ppg+dalia.zip"
TEMPLE_ZIP    = "data.zip"
WORKDIR       = "data_extracted"
# -------------------------------------------------------------


def unzip(z, dest):
    if not os.path.exists(z):
        print(f"  !! zip not found: {z}  (fix the path at the top of this file)")
        return None
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    return dest


def tree(root, limit=40):
    files = sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True))
    for f in files[:limit]:
        print("   ", os.path.relpath(f, root))
    if len(files) > limit:
        print(f"    ... (+{len(files) - limit} more)")


os.makedirs(WORKDIR, exist_ok=True)

print("=" * 70, "\nPHYSIONET  (wrist)\n", "=" * 70, sep="")
p = unzip(PHYSIONET_ZIP, os.path.join(WORKDIR, "physionet"))
if p:
    tree(p)

print("\n" + "=" * 70, "\nUCI  (PPG-DaLiA)\n", "=" * 70, sep="")
u = unzip(UCI_ZIP, os.path.join(WORKDIR, "uci"))
if u:
    pkls = sorted(glob.glob(os.path.join(u, "**", "*.pkl"), recursive=True))
    print(f"   found {len(pkls)} .pkl subject files")
    if pkls:
        with open(pkls[0], "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        print("   top-level keys :", list(d.keys()))
        print("   signal.wrist   :", list(d["signal"]["wrist"].keys()))
        print("   signal.chest   :", list(d["signal"]["chest"].keys()))
        import numpy as np
        print("   wrist BVP shape:", np.asarray(d["signal"]["wrist"]["BVP"]).shape)
        print("   label shape    :", np.asarray(d["label"]).shape)

print("\n" + "=" * 70, "\nTEMPLE  (data.zip)\n", "=" * 70, sep="")
t = unzip(TEMPLE_ZIP, os.path.join(WORKDIR, "temple"))
if t:
    tree(t)
    for csv in sorted(glob.glob(os.path.join(t, "**", "*.csv"), recursive=True))[:3]:
        print("\n   --- head of", os.path.relpath(csv, t), "---")
        df = pd.read_csv(csv, nrows=5)
        print("   columns:", list(df.columns))
        print(df.head().to_string())

print("\nDone. If the Temple columns above don't match what 03_/04_ expect, "
      "paste this output back.")
