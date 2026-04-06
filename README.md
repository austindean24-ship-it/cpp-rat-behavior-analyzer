# Three-Chamber CPP Rat Behavior Analyzer

This project is a beginner-friendly Python app for measuring how long **one rat** spends in the **left**, **center**, and **right** chambers of a standard three-chamber conditioned place preference (CPP) box.

The app is made for a fixed camera and one video at a time. You upload a video, draw the chamber layout on the first frame, and the software tracks the rat body centroid through the video.

## What This Project Can Do

- Upload one video
- Show the first frame
- Let you define the 3 chambers
- Track the rat centroid frame by frame
- Assign each frame to a chamber
- Report time spent in each chamber in seconds and percent
- Export CSV files
- Export an annotated MP4 with the centroid and chamber labels drawn on top
- Report simple quality-control warnings when tracking looks unstable
- Generate a synthetic practice video so you can test the pipeline without real animal data

## Important Assumptions

- One rat only
- Camera does not move
- Apparatus does not move
- Video is top-down or near top-down
- Chamber occupancy is based on the rat **body centroid**
- Best results usually come from a clear arena floor with moderate contrast between the rat and the background

## Easiest Way To Use It

The easiest chamber setup is:

1. Draw **one rectangle around the full apparatus**
2. Let the app split that box into **left / center / right**

That is the recommended mode for most users.

## New Practical Option For Chamber Scoring

The app now lets you choose what point should represent the rat when deciding chamber occupancy:

- `Head-and-shoulders proxy`
  Best choice when your lab rule is based on the front of the rat entering a chamber.
- `Smoothed body centroid`
  Good general-purpose option if you want a stable center-of-body rule.
- `Raw body centroid`
  The simplest center-of-body rule with no smoothing.

Important note:

- The app still assigns **one chamber per frame only**
- It does **not** double-count a frame into two chambers
- If your total time seems longer than the true video length, the most likely reason is incorrect video FPS metadata, so the app also includes a manual FPS override

## Project Files

Here is what each main file is for, in plain English:

- `app.py`
  The main Streamlit app. This is the file you run to open the local interface in your browser.
- `tracker.py`
  The rat tracking code. This is where the software finds the moving rat and estimates its centroid.
- `regions.py`
  The chamber and region logic. This file decides whether a centroid is in the left, center, right, boundary, or outside area.
- `analysis.py`
  Turns frame-by-frame results into summary tables such as total time per chamber and QC metrics.
- `io_utils.py`
  Handles video loading, metadata, CSV export, and annotated video export.
- `demo_generator.py`
  Creates a fake practice video with known chamber times.
- `validate_demo.py`
  Runs the synthetic demo and checks how close the measured results are to the expected results.
- `requirements.txt`
  The list of Python packages this project needs.
- `tests/`
  Small automated tests that check important parts of the code.

## Folder You Should Use

This project lives in:

```text
/Users/austindean/Documents/New project/cpp_behavior_app
```

If you are following the exact steps below, use that folder.

## Step-By-Step Setup For A Beginner

These instructions assume you are on a Mac, because that is the environment used to build the project.

### 1. Open Terminal

Do this:

1. Click the **Spotlight** search icon in the top-right corner of your Mac screen, or press `Command + Space`
2. Type `Terminal`
3. Click the **Terminal** app

You should now see a window with a blinking cursor.

Why you are doing this:

Terminal is the place where you paste the commands that start Python and the app.

### 2. Move Into The Project Folder

Copy and paste this exact command into Terminal, then press `Return`:

```bash
cd "/Users/austindean/Documents/New project/cpp_behavior_app"
```

What this does:

- `cd` means “go into this folder”
- The quotes are important because the folder name contains a space

What should happen next:

- Terminal should move into the project folder
- You usually will not see a big message, and that is normal

### 3. Create A Small Private Python Environment

Copy and paste this:

```bash
python3 -m venv .venv
```

What this does:

- Creates a local Python environment just for this project
- Keeps project packages separate from the rest of your computer

What should happen next:

- A new hidden folder named `.venv` will be created
- Again, Terminal may not print much, and that is okay

### 4. Turn That Environment On

Copy and paste this:

```bash
source .venv/bin/activate
```

What this does:

- Turns on the project’s private Python environment

How to tell it worked:

- You should usually see `(.venv)` at the start of the Terminal line

### 5. Install The Required Python Packages

Copy and paste this:

```bash
pip install -r requirements.txt
```

What this does:

- Installs Streamlit, OpenCV, NumPy, pandas, pytest, and the drawing widget used by the app

What should happen next:

- You will see lots of text scroll by
- At the end, you should see a success message

### 6. Start The App

Copy and paste this:

```bash
streamlit run app.py
```

What this does:

- Starts the local web app on your computer

What should happen next:

- After a few seconds, a browser tab should open automatically
- If it does not open automatically, Terminal will show a local web address, usually something like:

```text
http://localhost:8501
```

- If needed, copy that address and paste it into your browser

## What The App Should Look Like When It Works

You should see a page with:

- the title **Three-Chamber CPP Rat Behavior Analyzer**
- a video upload area
- a button to create a demo video for practice
- a drawing area that shows the first video frame
- a **Run analysis** button
- result tables after analysis finishes

## Beginner Walkthrough: Full Analysis

This is the simplest first run.

### 1. Upload A Video

Inside the app:

1. Click **Browse files**
2. Pick your CPP video
3. Wait a moment while the first frame loads

If you do not want to use your real data yet:

1. Click **Create demo video for practice**
2. The app will load a fake video automatically

### 2. Define The Three Chambers

Recommended method:

1. Leave the chamber method on **Split one arena box into left / center / right (Recommended)**
2. On the first frame, click and drag one rectangle around the entire CPP apparatus
3. Release the mouse

What should happen next:

- The app should show a preview with labels for **LEFT**, **CENTER**, and **RIGHT**

Manual polygon method:

1. Choose **Draw 3 chamber rectangles manually (Easy)** if you want separate chamber boxes without the automatic thirds split
2. Draw one rectangle for the left chamber
3. Draw one rectangle for the center chamber
4. Draw one rectangle for the right chamber

Advanced polygon method:

1. Choose **Draw 3 chamber polygons manually (Advanced)**
2. Draw exactly 3 shapes, one for each chamber
3. The app sorts them from left to right automatically

If the polygon method feels awkward, use the rectangle method. It is much easier.

### 3. Run The Tracking

1. Leave the default settings alone for your first test
2. If your lab defines chamber entry by the **head and shoulders**, choose **Head-and-shoulders proxy**
3. If the video file reports the wrong duration, turn on **Use a manual FPS value for timing**
4. Click **Run analysis**

What happens during this step:

- The app estimates a background
- It finds the rat in each frame
- It calculates the centroid
- It assigns that centroid to a chamber
- It builds the summary tables and output files

### 4. Read The Output

After the app finishes, you will see:

- a summary table
- a quality-control table
- optional warnings
- an annotated video preview if you chose that option

How to read the main table:

- `left`, `center`, `right` rows tell you how many frames and seconds were assigned to each chamber
- `percent_of_video` tells you what fraction of the full video was spent there
- `boundary` is used only if you set a neutral boundary margin larger than 0
- `outside` means the centroid landed outside the drawn arena
- `missing` means the app could not confidently estimate a centroid for that frame

How chamber time is computed:

- The app counts frames in each chamber
- Then it divides by the video FPS
- Example: 300 frames in the left chamber at 30 FPS = 10 seconds

### 5. Export CSV Or Annotated Video

At the bottom of the results section, click:

- **Download summary CSV**
- **Download per-frame CSV**
- **Download QC CSV**
- **Download raw tracking CSV**
- **Download annotated MP4** if you created the annotated video

What each export means:

- `summary.csv`
  Final chamber totals
- `per_frame_assignments.csv`
  One row per frame, including centroid and chamber assignment
- `qc_metrics.csv`
  Quality-control measurements
- `tracking_raw.csv`
  Raw tracking output before chamber assignment
- `annotated_output.mp4`
  The video with overlays drawn on top

## Extra Scripts

### Make A Synthetic Practice Video

If you want to generate a fake test video from Terminal instead of the app:

```bash
python demo_generator.py
```

This project mainly expects you to generate the demo from the app or use the validation script below.

### Run The Validation Script

This creates a synthetic video, analyzes it, and compares expected chamber times against measured chamber times.

From the project folder:

```bash
python validate_demo.py
```

What you should see:

- a message saying validation finished
- the path to the demo video
- the path to the comparison CSV
- the maximum timing error in seconds and percent

## Run The Tests

If you want to run the automated tests:

1. Make sure your virtual environment is active
2. From the project folder, paste:

```bash
pytest
```

What this does:

- Checks chamber assignment logic
- Checks time-summary logic
- Checks the synthetic end-to-end validation

## Common Mistakes And How To Fix Them

### Problem: `command not found: streamlit`

What it means:

- The Python environment is probably not active yet

How to fix it:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Then try:

```bash
streamlit run app.py
```

### Problem: `No module named ...`

What it means:

- One or more packages were not installed in the project environment

How to fix it:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Problem: The app opens, but the tracking looks wrong

Try these fixes in order:

1. Make sure the video is truly fixed-camera
2. Make sure the chamber rectangle tightly surrounds the apparatus
3. Try the easiest box-split method before manual polygons
4. Raise **Minimum contour area** a little if dust or reflections are being tracked
5. Lower **Maximum expected centroid jump per frame** if the centroid jumps unrealistically
6. Check the QC table for many low-confidence frames

### Problem: The rat disappears for some frames

What it means:

- The tracker may be losing contrast because of shadows, reflections, or lighting changes

What to try:

1. Use a cleaner arena box drawing
2. Try a better-lit video if possible
3. Increase contrast in the recorded experiment setup for future videos

### Problem: The browser page does not open automatically

How to fix it:

1. Look in Terminal for a local address such as `http://localhost:8501`
2. Copy it
3. Paste it into Safari, Chrome, or Firefox

## Quality-Control Metrics Explained Simply

- `tracking_success_rate_percent`
  How often the app had some centroid estimate available
- `direct_detection_rate_percent`
  How often the rat was directly detected without relying on fallback logic
- `missing_centroid_frames`
  Frames where no centroid could be assigned
- `low_confidence_frames`
  Frames where the app was less certain than usual
- `mean_contour_area_px`
  Average size of the tracked blob in pixels
- `boundary_frames`
  Frames very close to shared chamber borders
- `mean_centroid_jump_px`
  Average frame-to-frame movement of the smoothed centroid

## Boundary Rule

The app uses a simple documented rule:

- If the centroid falls exactly on a shared border and the neutral boundary margin is `0`, the earlier chamber wins in left-to-right order
- In practice, that means `left` wins over `center`, and `center` wins over `right`
- If you set a neutral boundary margin larger than `0`, points close to more than one chamber can be labeled as `boundary`

## Known Limitations

- This MVP is designed for one rat only
- It does not use deep learning
- Very strong shadows or reflections can still confuse the tracker
- If the rat and floor have very similar brightness, tracking can become unstable
- Manual correction of bad frames is not included yet
- Batch processing is not included yet

## Ideas For Future Improvements

- Batch processing of many videos
- Automatic apparatus detection
- Better tracking diagnostics
- Occupancy heatmaps
- Chamber entry counts
- Latency to first chamber entry
- Manual correction mode
- Better support for more irregular chamber shapes

## Outputs You Should Expect To See

After a normal run, this folder is created automatically:

```text
runtime_data/results/<video_name_timestamp>/
```

Inside it, you should usually see files like:

- `summary.csv`
- `per_frame_assignments.csv`
- `qc_metrics.csv`
- `tracking_raw.csv`
- `warnings.txt`
- `annotated_output.mp4` if you chose that option

## If You Want The Fastest Possible First Test

Do these exact steps:

1. Open Terminal
2. Run `cd "/Users/austindean/Documents/New project/cpp_behavior_app"`
3. Run `python3 -m venv .venv`
4. Run `source .venv/bin/activate`
5. Run `pip install -r requirements.txt`
6. Run `streamlit run app.py`
7. In the app, click **Create demo video for practice**
8. Draw one rectangle around the full apparatus
9. Click **Run analysis**
10. Check the summary table and annotated video

That is the easiest end-to-end test.
