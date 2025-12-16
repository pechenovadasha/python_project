# Eye Detection Toolkit

Scripts for finding pupil centers and corneal glints in eye image sequences. The code reads bright/dark frame pairs, computes pupil and glint coordinates, writes them to CSV/JSON, and builds basic accuracy plots.

## Files and Purpose
- `reading_files.py` — entry point. Scans folders with BMP frames, reads them in bright/dark pairs, runs processing, and prints per-folder stats.
- `processing.py` — wrapper around detection: calls the tracker, collects errors, saves coordinates to CSV.
- `center_detection.py` — main pupil tracker. Normalizes frames, finds contours, picks two largest, searches glints for each, supports ROI and multithreaded left/right processing.
- `glints_detection_research.py` — helpers to detect a glint on the bright frame (intensity threshold, closest glint to pupil center) and to compute contour centers.
- `reference_centers.py` — refines contours via a local ROI that suppresses glare, fits an ellipse to get a reference center.
- `centers_analisis.py` — saves coordinates to CSV, computes RMS error, prints summary, and plots coordinate traces.
- `json_analisis.py` — stores measured pupil diameters to JSON, runs stats (mean, variance, CV, synchronization), and saves plots.
- `install_mac.sh` — installs Python dependencies via `pip` (macOS oriented).
- `dataset/` — sample BMP set; filenames must be in correct order so adjacent files form a “bright/dark” pair.

## Installation
Requires Python 3.10+ and `pip`. Quick macOS setup:
```bash
chmod +x install_mac.sh
./install_mac.sh
```
Or install manually:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install opencv-python==4.11.0.86 numpy==1.24.3 pandas matplotlib pyautogui Pillow scipy
```

## Data Preparation
- Place BMP frames into folders inside `dataset/` or another directory.
- File order matters: `reading_files.py` processes them sequentially and pairs `(bright, dark)` frames.
- Create folders for outputs:
```bash
mkdir -p results statistics
```

## Run Processing
By default `reading_files.py` uses `folders = ['dataset']`.
```bash
python3 reading_files.py
```
The script subtracts bright and dark frames, finds pupil/glint centers, writes CSV to `results/<folder>.csv`, and prints how many pairs failed.

### Minimal example for one frame pair
```bash
python3 - <<'PY'
import cv2
from processing import recieve_centers
bright = cv2.imread('dataset/dariap0210_100_1_0.bmp')
dark = cv2.imread('dataset/dariap0210_100_1_1.bmp')
diff = cv2.subtract(bright, dark)
print(recieve_centers(diff, dark, bright, 'demo'))
PY
```

## Analytics and Visualization
- Compute RMS errors from a saved CSV:
```bash
python3 - <<'PY'
from centers_analisis import calculate_simple_rms
print(calculate_simple_rms('results/dataset.csv'))
PY
```
- Plot coordinate traces (vertical dashed lines mark frame blocks):
```bash
python3 -c "from centers_analisis import plot_eyes_separate; plot_eyes_separate('results/dataset.csv')"
```
- Analyze pupil diameters and save JSON/PNG:
```bash
python3 - <<'PY'
from json_analisis import save_to_json
save_to_json([20, 21, 20.5], [19.5, 20, 20.1])
PY
```

## Notes
- The detector expects two pupils per frame; if fewer contours or low contrast are found, points are treated as missing.
- ROI activates after a few frames and speeds up processing, so early frames can have temporary misses.
