# HOLOEYE SLM + camera acquisition

`acquire_slm_camera.py` reproduces the acquisition recipe from pages 3-5 of
`sciadv.adg4671_sm.pdf`:

- HOLOEYE LETO-3 phase resolution: 1080 x 1920.
- Vertical fringe lines with slope `4*pi/3` radians per pixel; phase varies
  left-to-right along the SLM x-axis.
- Random sums of the first 15 RMS-normalized Zernike modes, with independent
  Gaussian coefficients having a standard deviation of 5 radians.
- A 256 x 256 modulation cropped to 144 x 256, nearest-neighbor upsampled to
  the SLM, added to the fringe, and wrapped to `2*pi`.
- One full camera frame saved for every displayed pattern.
- Grasshopper3 camera defaults of 100 ms exposure, gain 0 dB, gamma 1, and
  unpacked `Mono16`.
- Paper-compatible downscaling by 3.57 followed by a centered 256 x 256 crop.

The script supports HOLOEYE Display SDK 4.x (`HEDS`) and the older
`holoeye.slmdisplaysdk` interface. Camera control currently defaults to the
original FLIR/Point Grey Grasshopper3 through Spinnaker (`PySpin`). Player One
support remains in the script for later use with `--camera-backend playerone`.

## 1. Validate without hardware

In the Windows VS Code terminal, create a local environment and install the
normal Python dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, run **Python: Select Interpreter** from the Command Palette and
choose `.venv`. Activation is optional; the commands below assume the selected
environment provides `python`. Validate without either hardware device:

```powershell
python acquire_slm_camera.py --dry-run --num-patterns 3 --output-format npy
python -m unittest -v test_acquire_slm_camera.py test_process_measurements.py test_calibrate_slm_camera.py
```

The dry run generates realistic-sized simulated 16-bit frames and exercises
the same pattern, preprocessing, metadata, and file-writing path as hardware
mode. Keep the explicit test filenames in the second command: broad unittest
discovery also imports the copied vendor `HEDS` package, which expects its
Windows SDK environment variable and therefore fails on macOS.

### Test only the old camera

With SpinView closed and the Grasshopper3 connected, capture one frame without
opening the HOLOEYE SDK or SLM:

```powershell
python camera_smoke_test.py `
  --camera-serial YOUR_CAMERA_SERIAL `
  --exposure-ms 100 `
  --gain-db 0 `
  --pixel-format Mono16
```

If only one Spinnaker camera is connected, omit `--camera-serial`. The script
prints the frame shape, data type, minimum, maximum, and mean, then saves both
`data/camera_test_TIMESTAMP.npy` and `.png`. The NPY contains the unchanged raw
camera values. The PNG is an 8-bit preview scaled between the frame's 1st and
99th percentiles for convenient viewing. For a color camera, use the unpacked
16-bit Bayer format reported by SpinView, such as `BayerRG16`; its preview is
still the raw Bayer mosaic, not a demosaiced color image.
If both percentiles coincide (for example, a tiny aperture on a dark field),
the preview falls back to the full intensity range so sparse signals stay visible.

## 2. Install vendor software on the acquisition PC

1. Install the HOLOEYE SLM Display SDK supplied through the HOLOEYE customer
   site. The included v4 manual says the Python Convenience API is the `HEDS`
   folder inside the SDK's `examples` directory. Either copy that `HEDS` folder
   beside `acquire_slm_camera.py`, or pass the parent directory using
   `--heds-examples-dir`. The `HEDS/detect_heds_module_path.py` helper then uses
   the installer-created environment variable to locate the lower-level
   `api/python` binding. The script defaults to HEDS API 4.2, matching the
   supplied v14 manual; pass `--heds-api-version 4.1` or `4.0` for an older SDK.
2. Install a FLIR/Teledyne Spinnaker SDK version that supports the connected
   Grasshopper3, including its matching `PySpin` package. Run SpinView first to
   confirm that the camera is visible and to note its serial number.
3. Install this script's normal Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

The supplied HOLOEYE desktop Display SDK is Windows-targeted. Run hardware
acquisition on the Windows computer driving the SLM, not on this Mac.

Before running Python, confirm the SDK manual's display requirements:

- Use Windows 10 or 11 and configure the SLM as an extended desktop display.
- Turn off Windows Night Light and other color-changing/display power-saving
  features.
- Run one of the SDK's supplied Python examples first. The vendor states that
  these examples are release-tested and are the recommended project starting
  point.
- Note the HOLOEYE SDK `examples` directory and Grasshopper3 serial number.

## 3. Supply the HOLOEYE SDK path on Windows

`--heds-examples-dir` must point to the SDK's `examples` directory. That
directory must contain an `HEDS` subdirectory. Do not point the option directly
to `examples\HEDS`, the SDK root, or `api\python`.

For an SDK folder on the Windows Desktop:

```powershell
$HedsExamples = "$env:USERPROFILE\Desktop\SLM Display SDK (Python) v4.2.1\examples"
Test-Path "$HedsExamples\HEDS"
```

For the standard Program Files installation:

```powershell
$HedsExamples = "C:\Program Files\HOLOEYE Photonics\SLM Display SDK (Python) v4.2.1\examples"
Test-Path "$HedsExamples\HEDS"
```

`Test-Path` must return `True`. Supply the validated value to the script:

```powershell
python acquire_slm_camera.py `
  --heds-examples-dir "$HedsExamples" `
  --slm-preselect "index:0" `
  --num-patterns 3 `
  --camera-serial YOUR_CAMERA_SERIAL `
  --output-dir data\test_run
```

To avoid supplying the option every time, save it as a user environment
variable:

```powershell
[Environment]::SetEnvironmentVariable(
  "HEDS_EXAMPLES_DIR",
  $HedsExamples,
  "User"
)
```

Restart PowerShell after setting the variable. The script can then find HEDS
without `--heds-examples-dir`. As another supported option, copy the complete
`HEDS` directory from the SDK's `examples` folder beside
`acquire_slm_camera.py`.

## 4. Acquire data

Close SpinView and any HOLOEYE example that has exclusive control of a device,
then start with a short, low-risk run:

```powershell
python acquire_slm_camera.py `
  --num-patterns 3 `
  --heds-examples-dir "$HedsExamples" `
  --slm-preselect "index:0" `
  --camera-serial YOUR_CAMERA_SERIAL `
  --output-dir data\test_run
```

Inspect the captured intensities and pattern changes. Then acquire the desired
sequence, for example:

```powershell
python acquire_slm_camera.py `
  --num-patterns 100 `
  --seed 1 `
  --exposure-ms 100 `
  --gain-db 0 `
  --gamma 1 `
  --pixel-format Mono16 `
  --settle-ms 150 `
  --wavelength-nm 532 `
  --camera-serial YOUR_CAMERA_SERIAL `
  --output-dir data/sample_001
```

The paper's 3.57 magnification is specific to its optical train. Perform the
aperture/fringe-shift calibration described on pages 4-5 for your setup and
pass the measured value as `--magnification`. Likewise, choose an exposure in
the paper's 90-120 ms range only after checking that your sample is not
saturating. Set `--wavelength-nm` to your actual laser wavelength because
HOLOEYE uses it for phase modulation; do not retain 532 nm if your laser is
different. Keep the beam blocked while testing software/device selection.

For the current 1920 x 1200 setup, the commands in `RUN_COMMANDS.txt` use
horizontal fringe lines and retain both zero and first orders in the same
camera frame. Use `--capture-fringe-first --no-neuws-processing` to capture
the pure steering fringe first, then add each random phase pattern to that
same fringe and capture another full frame. The SLM and camera stay open
throughout the sequence. Each display update uses `--settle-ms` and
`--discard-frames`, including the initial fringe reference. Ensure the camera
ROI is set to the full sensor and both orders lie within its field of view.

The fringe reference is extra: `--num-patterns 3 --capture-fringe-first` saves
four frames total before any steering retries (one fringe reference and three
random-plus-fringe frames). Each adjustment or recapture adds another reference.
The random pattern seed and numbering are unchanged by the initial reference.
The full reference is saved as `reference/fringe_only.npy` plus a PNG preview;
the following frames use the usual `raw/` and `previews/` directories.

The commands also enable `--confirm-fringe-preview`. After saving the fringe
reference, the script opens `reference/fringe_only.png` in the default image
viewer and waits for terminal feedback while the fringe remains on the SLM:

```text
Are zero and first orders separated? Proceed with random phase patterns? [Y/N, P pixels, R, H, V, C]:
```

At this prompt you can adjust the steering and capture another preview:

- `P 20`: set a 20-pixel fringe period; `P -20` selects the opposite ramp slope.
  The signed period must be finite and have an absolute value of at least 2 pixels.
- `R`: reverse the current ramp slope.
- `H` / `V`: select horizontal / vertical fringe lines.
- `C`: capture a fresh fringe-only image, for example after a physical adjustment.
- `Y`: accept the current fringe and begin the random-plus-fringe measurements.
- `N`: stop and close the devices, retaining all captured references.

Each adjustment applies the fringe, waits for settling, discards buffered frames,
saves a new full-frame NPY/PNG, and opens that preview for another decision.
The printed sampled period accounts for equivalent ramps on the pixel grid;
its sign describes ramp direction. Confirm the actual steering in the camera
image. The initial slope still defaults to `4*pi/3`; to choose initial spacing
directly, pass `--fringe-period-pixels 20` (or a negative period). This option
is mutually exclusive with `--fringe-slope-rad-per-pixel`.

Random patterns use the accepted fringe with their original seeded coefficient
sequence. `manifest.json` records the final configuration, every reference and
its feedback in `fringe_references`, and the latest reference in `fringe_reference`.
The first preview is `reference/fringe_only.png`; retries have unique names such
as `reference/fringe_only_0002.png`, so previous captures are retained.
Invalid/blank commands repeat the prompt; unavailable terminal input stops the
run. If the viewer cannot open automatically, open the printed PNG path manually.
`--confirm-fringe-preview` enables `--capture-fringe-first` automatically when
used alone.

To select only the first order later, inspect the full-frame PNG and supply
its raw-camera `(x, y)` position with `--crop-center-x` and `--crop-center-y`,
omitting `--no-neuws-processing`. Full raw frames are still retained.

The current run commands use `--output-format npy` and omit
`--save-full-patterns` to reduce disk use. Camera NPY/PNG measurements, small
phase-modulation arrays, coefficients, and steering settings are retained.
The full phase array for each fringe-only reference is also retained.

For a color Grasshopper3, select an unpacked 16-bit Bayer format supported by
the camera, such as `BayerRG16`, instead of `Mono16`. Check the supported pixel
formats in SpinView. The script discards `--discard-frames` buffered frames
after each SLM change before saving the next frame.

## 5. Player One camera later

Player One SDK 3.10.1 support is retained but is not selected by default. When
you are ready to change cameras, supply the SDK root and backend explicitly:

```powershell
$PlayerOneSdk = "$env:USERPROFILE\Desktop\PlayerOne_Camera_SDK_Windows_V3.10.1"

python acquire_slm_camera.py `
  --camera-backend playerone `
  --playerone-sdk-dir "$PlayerOneSdk" `
  --pixel-format RAW16 `
  --gain 0 `
  --num-patterns 3 `
  --output-dir data\playerone_test
```

The Player One adapter uses SDK Snap Mode and loads the correct DLL from
`lib\x64` or `lib\x86`. It can also use the `PLAYERONE_SDK_DIR` environment
variable or automatically find a copied `PlayerOne_Camera_SDK*` folder beside
the script.

## 6. Extract a camera region with its matching phase

Use `process_measurements.py` after acquisition. It opens no hardware and reads
the unchanged raw NPY measurements rather than contrast-scaled PNGs. Draw a
rectangle around the steered image on the reference preview, then press Enter:

```powershell
python process_measurements.py datameasurement_aa --output-dir processed\measurement_aa_selected --include-full-phase
```

For `datameasurement_aa`, the phase-varying left-hand image is the candidate
steered component. A suggested camera crop is `(x=300, y=200, width=900,
height=1100)`. Confirm the outlined region in `roi_preview.png`:

```powershell
python process_measurements.py datameasurement_aa --roi 300 200 900 1100 --output-dir processed\measurement_aa_steered --include-full-phase
```

This produces a new dataset containing `pairs/frame_0001.npz`, camera PNGs in
`previews/`, wrapped random-phase PNGs in `phase_previews/`, `roi.json`, and a
manifest mapping every output sample to its original pattern index. The camera
crop keeps the original dtype and values, without resizing or normalization.

Each compressed pair contains:

- `image`: the cropped camera measurement.
- `phase_modulation_rad`: its original 144 x 256 random phase, before the fringe.
- `phase_slm_rad`: the full displayed random-plus-fringe phase when
  `--include-full-phase` is enabled; saved full masks are preferred, otherwise
  the phase is reconstructed from the recorded coefficients and settings.
- `zernike_coefficients_rad`, `fringe_slope_rad_per_pixel`,
  `fringe_orientation`, `slm_shape`, `camera_roi_xywh`, and `source_pattern_index`.

The phase stays in SLM coordinates. Applying the camera ROI to the phase array
would require optical registration and is not part of this processing. Omit
`--include-full-phase` for smaller output files; the small modulation and
coefficients/steering needed to reconstruct the mask are still included.

```python
import numpy as np

with np.load("processed/measurement_aa_steered/pairs/frame_0001.npz", allow_pickle=False) as sample:
    image = sample["image"]
    phase = sample["phase_slm_rad"]
```

Dark/partial measurements are flagged in the manifest and retained by default.
Use `--exclude-indices 4 5 6 7 34 60 61 62 89 90` only when you explicitly want
to omit the previously identified weak frames. Their phases are omitted with
them, and output numbering stays contiguous while preserving source indices.
Add `--export-mat` to also write cropped `imsdata` / modulation `proj_sim` MAT
pairs. Crops are not automatically resampled to the paper's 256 x 256 size.
All offline commands are also included as single lines in `RUN_COMMANDS.txt`.

## 7. Measure SLM/camera Fourier sampling

`calibrate_slm_camera.py` implements the supplement's aperture/ramp calibration.
Replace the object with a small aperture, remove the aberrating sample, and
focus the aperture image. Keep the optical arrangement, camera ROI/binning,
and wavelength fixed between calibration and acquisition. Run the hardware
calibration command in `RUN_COMMANDS.txt` on the Windows acquisition PC.

The command starts with the 20-pixel horizontal fringe used for the uploaded
measurements. On the first camera image, draw a rectangle containing only the
steered aperture, excluding the static zero order. Leave background margin
around the spot (at least 24 pixels from its center), then press Enter.
Ten ramp positions are captured: the carrier, signed offsets on both SLM axes,
and a return to the carrier. Three raw frames are retained at each position.
The script checks weak/saturated/edge detections and fits a robust affine map:

```text
camera_xy = camera_origin + B @ [fx, fy]
```

`fx` and `fy` are cycles per physical SLM pixel; camera coordinates are raw
sensor pixels. The 2 x 2 matrix captures axis rotation/reflection and unequal
scaling. Inspect `calibration_review.png` and answer `Y` only when the marked
positions correspond to the same steered aperture. A static zero-order spot
cannot pass the two-axis motion checks. The fit also checks center-return drift.
Start with 10 ms exposure and adjust as needed to keep the aperture unsaturated;
`--saturation-level` can specify the actual maximum raw camera value if the
effective ADC range differs from the uint16 container's maximum of 65535.

Outputs include every raw capture, ramp metadata, median images, phase PNGs,
the review plot, `detections.json` (including rejected detections), and
`calibration.json`. The first capture is used to estimate required free disk
space before the scan continues (roughly 400 MB for 30 raw 2000 x 1500 frames,
median images, and previews). Only a passing, operator-approved,
non-synthetic result has `usable_for_hardware: true`. A simulation validates
the software only; no optical scale is inferred from the old random-pattern
dataset, which has only one steering carrier.

The result stores the raw frequency-to-camera map and its interpretation on
a square FFT grid. For a grid with N columns/rows and an active pupil with
`pupil_rows` rows, the predicted FFT-bin shift per ramp frequency is:

```text
slm_x: slm_width * fx
slm_y: (N * slm_height / pupil_rows) * fy
```

The existing generator/captures use 144 modulation rows, even with a 1920 x
1200 SLM, so the hardware command explicitly uses `--pupil-rows 144`.
An aspect-matched full-area 1920 x 1200 pupil on a 256 grid would have 160
active rows. This model choice changes the normalized FFT sampling, while
the measured raw optical matrix B is unchanged. The report therefore retains
B instead of forcing the two axes into one assumed magnification number.
It does not calibrate the SLM LUT, beam footprint, or direct image-plane
registration, and does not automatically resize the existing image/phase pairs.

If a spot leaves the search window, reduce `--step-cycles-per-pixel` or adjust
`--tracking-radius`, while keeping the static zero order outside the search.
Use `--analyze-only` to refit saved captures without opening hardware, optionally
with a corrected `--roi`. `--skip-review` saves an unapproved result for later
inspection. Each run/refit needs a new output directory.

## Output

Each run contains:

```text
sample_001/
  manifest.json
  raw/frame_0001.npy
  previews/frame_0001.png
  previews/processed_0001.png
  processed/frame_0001.npy
  patterns/phase_modulation_0001.npy
  neuws_mat/SLM_raw1.mat
  neuws_mat/SLM_sim1.mat
```

The full sensor frame is always retained unchanged in `raw/`. `previews/`
contains viewable 8-bit PNGs scaled between each frame's 1st and 99th
percentiles; these previews are for inspection rather than quantitative use.
If those limits coincide, the full intensity range is used instead.
`processed/` contains the 3.57x downscaled 256 x 256 measurement. By default
it is centered on the sensor. To select a displaced first diffraction order
instead of the static zero order, inspect `previews/frame_0001.png`, determine
the first-order center in raw camera pixels, and pass `--crop-center-x X` and
`--crop-center-y Y`. `previews/processed_0001.png` shows the selected crop.
The MAT files use the variable and filename conventions expected by the public NeuWS reconstruction code:
`imsdata` in `SLM_rawN.mat` and `proj_sim` in `SLM_simN.mat`.

`manifest.json` records all settings, Zernike coefficients, file paths, camera
frame identifiers, and display/capture timestamps. It is rewritten after each
successful frame, so completed data remains described if acquisition stops.
Camera measurements are recorded in the manifest before writing a large full
SLM phase export. If that export fails, its `full_pattern_path` remains null
and a `full_pattern_save_error` is recorded when disk space permits.
When `--capture-fringe-first` is used, the `fringe_reference` entry records the
initial reference files and timestamps. `frames_completed` counts only the
subsequent frames requested by `--num-patterns`.

Useful options:

- `--save-full-patterns`: retain every 1080 x 1920 displayed phase array.
- `--capture-fringe-first`: capture a full-frame pure fringe reference before
  the requested random-plus-fringe measurements; the reference phase is also saved.
- `--confirm-fringe-preview`: open the fringe reference and require a `Y/N`
  response before random phase patterns, with steering and recapture controls;
  also enables the reference capture.
- `--fringe-period-pixels 20`: choose initial fringe spacing using a signed
  period in SLM pixels instead of `--fringe-slope-rad-per-pixel`.
- `--no-neuws-processing`: retain raw frames but skip scaling/cropping.
- `--output-format npy`: avoid the SciPy MAT-file dependency.
- `--discard-frames N`: discard buffered PySpin frames after each SLM update.
- `--slm-preselect STRING`: select an SLM in HEDS 4.x multi-SLM setups.
- `--heds-examples-dir PATH`: locate the installed SDK's `examples/HEDS` API.
- `--heds-api-version 4.1`: match an older installed HEDS SDK.
- `--playerone-sdk-dir PATH`: locate the Player One SDK root or `python` folder.
- `--camera-backend playerone`: switch from the default PySpin camera.
- `--wavelength-nm 532`: select the wavelength used by HOLOEYE's phase lookup.
- `--crop-center-x X --crop-center-y Y`: center processed output on the
  steered diffraction order using raw-camera pixel coordinates.
- `--fringe-orientation vertical`: use vertical fringe lines (the default),
  with phase varying along x. Use `horizontal` to vary phase along y instead.

The reference does not specify its Zernike sign/order convention, camera color
model, or the coordinate units attached to the stated fringe slope. This
implementation interprets the slope as radians per SLM pixel, records its
convention and coefficients in the manifest, and keeps those choices isolated
in `zernike_basis()`, `--fringe-slope-rad-per-pixel`, and `--pixel-format` so
they can be adjusted without changing the acquisition loop.

### Storage write errors

`OSError: ... requested and 0 written` at `np.save` means the output file
could not be written. Check available space on the acquisition PC:

```powershell
Get-PSDrive -PSProvider FileSystem
```

At 1920 x 1200, each full float32 phase array contains 2,304,000 values and
needs 9,216,000 bytes (about 9.2 MB), plus the NPY header. One hundred full
phase files alone need about 0.92 GB, before camera images and other exports.
Omit `--save-full-patterns` and use `--output-format npy` to avoid optional
full phase exports and duplicate MAT measurements. Select a new output folder
on a drive with sufficient space; existing run directories are not overwritten.
If space is available, check write permissions, quotas, and drive errors.

NPY writes use a temporary file and rename it only after the write succeeds.
Failed NPY temporary files are removed when possible. Array writes check free
space and require 1 MiB of working headroom, but concurrent disk activity can
still exhaust the drive. Saved camera measurements are checkpointed before
the full phase export so a failure at that step retains their manifest entry.
# cataract
