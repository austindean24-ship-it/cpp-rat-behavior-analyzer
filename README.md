# Three-Chamber CPP Rat Behavior Analyzer

Python and Streamlit app for analyzing single-rat behavior in a standard three-chamber conditioned place preference (CPP) apparatus from top-down or near top-down video.

The app is built for practical lab use: upload one video, draw the three chamber regions on the first frame, run tracking, and export chamber-time summaries, per-frame data, QC metrics, and an optional annotated video.

## Overview

This project uses a lightweight classical computer vision pipeline rather than a deep-learning model. Tracking is based on motion segmentation, contour filtering, and temporal smoothing. Chamber occupancy can be scored using either a body-centered point or a front-of-body proxy, depending on the lab's operational definition.

The current interface is intentionally opinionated:

- one video at a time
- one rat at a time
- fixed tracking settings in the UI for consistency across runs
- chamber drawing on a single first-frame image using three click-and-drag rectangles

## Key Features

- Streamlit web interface for local use
- First-frame preview for chamber calibration
- Three-chamber drawing workflow on one image
- Single-animal tracking with fallback logic for short misses
- Chamber assignment by frame
- Time spent in each chamber in seconds and percent
- Optional manual FPS override when video metadata is wrong
- CSV export for summary, per-frame assignments, QC metrics, and raw tracking
- Optional annotated MP4 export
- Synthetic demo video generator for practice and validation
- Automated tests and synthetic validation script

## Tracking Method

The tracker is designed as a practical MVP for fixed-camera CPP videos.

Core approach:

- background differencing
- frame-to-frame motion differencing
- morphological cleanup
- largest valid contour selection
- centroid estimation from contour moments
- light temporal smoothing
- short-gap carry-forward logic for brief missed detections

Important behavior:

- tracking is restricted to the exact drawn chamber regions
- motion outside the chamber polygons is ignored
- chamber assignment uses one point per frame only
- the app does not double-count a frame into two chambers

Available chamber-scoring modes:

- `Head-and-shoulders proxy`: recommended when chamber entry is defined by the front of the rat
- `Smoothed body centroid`: stable center-of-body scoring
- `Raw body centroid`: unsmoothed center-of-body scoring

## Assumptions

This project works best when these conditions are true:

- one rat only
- fixed camera
- stationary apparatus
- top-down or near top-down viewpoint
- clear chamber boundaries visible in the frame
- reasonable contrast between the rat and the arena floor

## Installation

### 1. Clone or download the repository

```bash
git clone https://github.com/austindean24-ship-it/cpp-rat-behavior-analyzer.git
cd cpp-rat-behavior-analyzer
```

If you already downloaded the project as a folder, open Terminal and `cd` into that folder instead.

### 2. Create a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Current Python dependencies are:

- `streamlit`
- `opencv-python-headless`
- `numpy`
- `pandas`
- `Pillow`
- `pytest`

## Run the App

From the project root:

```bash
streamlit run app.py
```

Streamlit should print a local address such as:

```text
http://localhost:8501
```

Open that address in your browser if it does not launch automatically.

## Typical Workflow

### 1. Load a video

- Upload a CPP session video, or
- click **Create demo video for practice**

### 2. Check timing

- Leave timing alone unless the reported video duration is clearly wrong
- If needed, enable the manual FPS override and enter the correct FPS

### 3. Define chambers

- Draw exactly three rectangles on the first-frame preview
- Draw one rectangle each for the left, center, and right chambers

### 4. Choose the scoring point

- Use `Head-and-shoulders proxy` if chamber entry should reflect the front of the rat
- Use one of the centroid options if your lab scores position from the body center

### 5. Run analysis

The app will:

- estimate the background
- track the rat frame by frame
- assign one chamber label per frame
- calculate summary and QC tables
- export CSV files
- optionally render an annotated video

If annotated video export is enabled, the tables and CSV downloads appear before the MP4 finishes rendering.

## Outputs

Each analysis run creates a results folder under:

```text
runtime_data/results/<video_name_timestamp>/
```

Typical output files:

- `summary.csv`
- `per_frame_assignments.csv`
- `qc_metrics.csv`
- `tracking_raw.csv`
- `warnings.txt`
- `annotated_output.mp4` if video export was enabled

### Output file descriptions

- `summary.csv`: total time and percent of video for each chamber label
- `per_frame_assignments.csv`: frame-by-frame chamber assignment and chosen scoring point
- `qc_metrics.csv`: tracking health metrics and warning counts
- `tracking_raw.csv`: raw tracking outputs before chamber labeling
- `warnings.txt`: plain-text QC warnings, if any

## How Chamber Time Is Computed

Chamber time is calculated from frame counts and FPS:

```text
seconds_in_chamber = frames_in_chamber / fps
```

Example:

```text
300 frames in left chamber at 30 FPS = 10 seconds
```

## Boundary Handling

The app uses a deterministic boundary rule.

- if the scoring point falls exactly on a shared border and the neutral boundary margin is `0`, the earlier chamber wins in left-to-right order
- in practice, `left` wins over `center`, and `center` wins over `right`
- if a neutral boundary margin greater than `0` is used, near-border points may be labeled as `boundary`

## Quality Control Metrics

The QC table includes:

- `tracking_success_rate_percent`
- `direct_detection_rate_percent`
- `missing_centroid_frames`
- `low_confidence_frames`
- `mean_contour_area_px`
- `boundary_frames`
- `mean_centroid_jump_px`

These metrics are mainly intended to flag runs where shadows, reflections, poor contrast, or weak chamber calibration may have affected the track.

## Demo and Validation

### Generate a synthetic demo video

You can create a synthetic CPP video directly from the app, or from the command line:

```bash
python demo_generator.py
```

The current demo is five minutes long and is meant for practice, smoke testing, and pipeline validation.

### Run the validation script

```bash
python validate_demo.py
```

This script:

- creates a synthetic video
- runs the full tracking and chamber-assignment pipeline
- compares expected chamber times with measured chamber times
- writes validation CSV outputs

## Run the Tests

From the project root:

```bash
pytest
```

The test suite covers:

- chamber assignment logic
- summary calculations
- canvas compatibility helpers
- cropped tracking behavior
- synthetic validation thresholds

## Project Layout

| File or Folder | Purpose |
| --- | --- |
| `app.py` | Main Streamlit interface |
| `tracker.py` | Tracking pipeline and tracking configuration |
| `regions.py` | Chamber geometry, ordering, and boundary logic |
| `analysis.py` | Summary tables, per-frame labels, and QC calculations |
| `io_utils.py` | Video IO, metadata, CSV export, annotated video export |
| `demo_generator.py` | Synthetic demo generation |
| `validate_demo.py` | End-to-end validation against synthetic truth |
| `canvas_utils.py` | Patched drawing canvas integration for local and hosted use |
| `tests/` | Automated tests |
| `vendor/drawable_canvas_build/` | Vendored canvas frontend build |
| `runtime_data/` | Uploaded videos, generated demos, and analysis outputs |

## Troubleshooting

### `command not found: streamlit`

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

Then reinstall dependencies if needed:

```bash
pip install -r requirements.txt
```

### `No module named ...`

The environment is active, but one or more packages are missing. Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Tracking looks wrong

Check the obvious failure points first:

- chamber rectangles do not match the apparatus closely
- the camera is not fully fixed
- the rat has poor contrast against the floor
- reflections or shadows are being mistaken for motion

The annotated video and QC table are the fastest ways to diagnose this.

### The rat is missing for some frames

This usually points to low contrast, reflections, strong shadows, or glare. Improving chamber calibration can help, but some videos may need cleaner acquisition conditions to track reliably.

### The reported video time looks wrong

If total time is inconsistent with the known session length, inspect the video FPS metadata. Use the manual FPS override in the app when the file reports the wrong timing.

## Known Limitations

- single-rat only
- fixed-camera only
- no manual correction workflow
- no batch processing in the current UI
- no deep-learning-based detector
- very strong shadows or reflections can still degrade tracking

## Future Work

- batch processing
- automatic apparatus detection
- richer tracking diagnostics
- occupancy heatmaps
- chamber entry counts
- latency-to-entry metrics
- manual correction mode
- broader support for irregular chamber geometries
