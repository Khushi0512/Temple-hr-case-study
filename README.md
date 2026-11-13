# Temple-hr-case-study
# Temple HR: Independent Analysis

A small, reproducible re-analysis of **Temple's open heart-rate validation dataset**, looking at three things their paper doesn't report: how each device tracks heart rate **while it's coming down** (recovery), how many seconds each device **lags** the reference, and how often each device puts you in the **wrong training zone**.

**_Everything here runs on publicly released data. I'm not affiliated with Temple, this started as a fan pressure-testing their open data, which they've publicly invited people to do._**

## Why these metrics

Temple report accuracy as mean error in bpm. That's fair, but it isn't what an athlete feels. You feel it when your heart rate is dropping after an interval and the watch lags, or when a few bpm near a zone boundary flips you into the wrong training zone. So I asked the data those two questions instead, using only the files Temple and two public datasets already publish.

## Sanity check first

Before reporting anything new, I rebuilt Temple's own headline number from their raw data. Their paper reports an overall MAPE of ~0.98%; my pipeline computes **0.93%** on the same data. Reproducing their published figure is the evidence that the pipeline is real: the new numbers below come out of that same code.

## Results

**Temple's own dataset (n = 43, each device's actual firmware HR vs taped Polar H10):**

| Device | Recovery error (decline MAPE) | Lag | Wrong-zone time | Wrong-zone (Zones 4–5) |
|---|---|---|---|---|
| **Temple** | **0.93%** | **+1 s** | **5.4%** | **6.2%** |
| Apple Watch S12 | 1.57% | +3 s | 10.0% | 8.7% |
| WHOOP MG | 2.00% | +1 s | 15.9% | 15.9% |
| Fitbit Air | 2.09% | +5 s | 13.9% | 11.0% |
| Garmin FR970 | 2.62% | +7 s | 14.9% | 11.5% |
| Oura Ring 5 | 3.76% | −3 s | 23.5% | 25.6% |

**External context (one dataset, raw-signal quality only):** on UCI PPG-DaLiA, a standard accelerometer-based motion-compensated extraction of wrist PPG still carries a median recovery error of **~14%** under everyday motion; a reminder that the wrist is a hard optical *site*, independent of any one device's firmware.

## Datasets (download these yourself — not redistributed here)

- **Temple heart-rate validation data** — from Temple's science page: https://temple.com/science-and-research (the "available here" data link in the heart-rate accuracy white paper).
- **UCI PPG-DaLiA** — https://archive.ics.uci.edu/dataset/495/ppg+dalia
- **PhysioNet Wrist PPG During Exercise** — https://physionet.org/content/wrist/1.0.0/

Extract each into a `data_extracted/` folder:

```
data_extracted/
  temple/      <- Temple's extracted data (contains temple_heart_rate_validation_data/)
  uci/         <- PPG-DaLiA (extract the inner data.zip too, so the .pkl files appear)
  physionet/   <- wrist-ppg-during-exercise-1.0.0/
```

## Setup

```
pip install wfdb scipy numpy pandas matplotlib
```

## Run (in this order)

```
python 00_inspect.py            # prints the structure of each dataset
python 03_temple_lag.py         # Temple: recovery error + lag, per device
python 04_temple_zone_error.py  # Temple: wrong-zone %, per device
python 02_uci_lag.py            # UCI: motion-compensated wrist-PPG recovery error
python 01_physionet_lag.py      # PhysioNet (see note below)
```

Each script prints its numbers and saves a figure. `hrlib.py` is the shared library the analysis scripts import; you don't run it directly.

## How the analysis works

**Reference vs device.** The reference heart rate is always derived from ECG; the taped Polar H10 chest strap in Temple's data, the ECG-derived ground truth in PPG-DaLiA, and the chest-ECG R-peaks in PhysioNet. The device heart rate is always the optical/PPG estimate. The reference is only ever used to *score* a device; it never enters the HR extraction itself.

**Heart rate from PPG.** For the external datasets I compute HR from the raw optical signal with a windowed spectral method: band-pass the PPG to the human-HR band (0.5–4 Hz), then for each 8-second window (hopping every 2 s) take the strongest spectral peak as the heart rate.

**Motion compensation (the DSP step).** A naive spectral peak fails during walking/running because step cadence lands in the same frequency band as heart rate, so the estimator locks onto the footstep rhythm instead of the pulse. To handle this, for each window I also compute the accelerometer spectrum (summed over axes, resampled onto the PPG grid), find the dominant motion/cadence frequency, and — only when real motion is actually present (the ACC peak is well above its own baseline) — null the PPG spectral bins at that cadence frequency and its second harmonic before picking the heart-rate peak. A short median filter then removes isolated spikes. This uses only the device's own PPG and accelerometer; it never sees the ECG. All of its parameters (8 s window, 0.5–4 Hz band, cadence tolerance of ~1.5 FFT bins, motion-presence threshold, median-filter width) are fixed physiological/DSP constants chosen before looking at any result.

**Recovery segments.** The "recovery" (decline) error is computed only over the stretches where the smoothed reference HR is falling; identified from the sign of the smoothed slope, with no threshold tuned to a result.

**Lag.** Reported as the integer-second cross-correlation offset that best aligns a device to the reference. A positive lag means the device reacts late.

**Training zones.** Zones use HRmax = 208 − 0.7 × age (the Tanaka formula), with age taken from each participant's published age band at the band midpoint. Each second is assigned a zone from the reference HR and from the device HR; "wrong-zone time" is the fraction of seconds where those two disagree.

## A note on PhysioNet

I include `01_physionet_lag.py` for transparency, but I don't use its number in the headline. PhysioNet's protocol is mostly walking and running, where step cadence sits on top of the heart rate in frequency — so a basic extraction can't separate them, and the walk/run records blow up. That's a real, well-known limitation of wrist PPG, not a bug, and I'd rather show it than hide it. The usable external number comes from the daily-activity dataset (UCI).

## Not affiliated with Temple

This is an independent analysis of publicly available data, shared in good faith.
