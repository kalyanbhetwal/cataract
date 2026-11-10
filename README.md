# HOLOEYE SLM + camera acquisition

This repository covers data collection, SLM/camera calibration, and offline
preprocessing. Generated data, server reconstruction tools, and transfer
utilities are kept locally and excluded from version control.

`acquire_slm_camera.py` defaults to replaying the NeuWS modulation arrays in
`data/neuws_dog_esophagus_phases/` beside the script. Install these phase files
using [the acquisition instructions below](#4-acquire-data); missing files stop
the run before hardware opens. Use `--phase-pattern-dir PATH` for another saved
sequence. The legacy local random generator requires `--generated-patterns`.
The acquisition recipe is based on pages 3-5 of `sciadv.adg4671_sm.pdf`:

- HOLOEYE LETO-3 phase resolution: 1080 x 1920.
- Vertical fringe lines with slope `4*pi/3` radians per pixel; phase varies
  left-to-right along the SLM x-axis.
- Random sums of the first 15 RMS-normalized Zernike modes, with independent
  Gaussian coefficients having a standard deviation of 5 radians.
- A 256 x 256 modulation cropped to 144 x 256, nearest-neighbor upsampled to
  the SLM, added to the fringe, and wrapped to `2*pi`.
- Ten full camera images averaged for every displayed pattern, with the
  individual images also saved by default.
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
mode. It needs the saved phase files, just like a hardware run. For a software
smoke test without those data assets, explicitly add `--generated-patterns`.
The unit tests use temporary patterns and do not need the data assets.
Keep the explicit test filenames in the second command: broad unittest
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

### Replay the NeuWS patterns by default — current 633 nm experiment

Use the actual `proj_sim` arrays from
`dog_esophagus_0.5diffuser/Zernike_SLM_data/SLM_sim1.mat` through `SLM_sim100.mat`.
The prepared, local-only folder is `data/neuws_dog_esophagus_phases/`. To move
it to the acquisition PC, copy `data/neuws_dog_esophagus_phases.zip` into that
checkout's `data/` directory, then extract it there:

```powershell
Expand-Archive data\neuws_dog_esophagus_phases.zip -DestinationPath data
```

This is the default directory, resolved beside the script even when launched
from a different working directory. Alternatively, point `--phase-pattern-dir` at the
`Zernike_SLM_data` directory. The data source is
[the NeuWS release](https://doi.org/10.5061/dryad.6t1g1jx42).
The folder and ZIP are data assets excluded from git; updating the scripts alone
does not install these phase files on another computer.
The complete local update bundle `data/neuws_replay_acquisition_update_20261110_v2.zip`
includes the updated scripts and this phase folder for the acquisition PC.

With the current 633 nm laser, 1920x1200 SLM, P15 horizontal steering and 20 ms
exposure, collect the fringe reference and 100 replayed patterns using:

```powershell
python acquire_slm_camera.py `
  --phase-pattern-dir data\neuws_dog_esophagus_phases `
  --num-patterns 100 --frames-per-pattern 10 --save-burst-frames `
  --capture-fringe-first --confirm-fringe-preview `
  --slm-width 1920 --slm-height 1200 `
  --fringe-orientation horizontal --fringe-period-pixels 15 `
  --wavelength-nm 633 --exposure-ms 20 --gain-db 0 --gamma 1 `
  --pixel-format Mono16 --settle-ms 1000 --discard-frames 2 `
  --slm-preselect "index:0" --camera-serial 20100892 `
  --no-neuws-processing --output-format npy `
  --output-dir data\measurement_633nm_replay_001
```

Choose a new output directory for each acquisition. The replay mode validates
every requested MAT file before opening hardware: `proj_sim` must be a finite,
real 144x256 array in radians. Patterns run in numeric order, starting at 1,
without cycling. `--num-patterns 3` uses the first three files. Steering retries
do not advance or reload the prevalidated sequence.

The saved modulation keeps the source values and precision, including nonzero
corners. It is nearest-neighbor resized to the configured SLM dimensions, added
to the accepted steering fringe, wrapped to `2*pi`, and converted to float32
for display. `--seed`, `--zernike-modes` and `--coefficient-std-rad` do not change
replayed patterns. Each manifest record includes the source filename, numeric
index and SHA256; unavailable generating coefficients are recorded as `null`.
Offline preprocessing retains this provenance and can reconstruct the full
display phase from the saved modulation and accepted fringe.

This command averages ten exposures and saves all individual frames for each
pattern and reference. It retains full camera images for later scale-and-crop
processing. Replaying the patterns does not establish a new 633 nm calibration
or make the 1920x1200 SLM aspect ratio identical to the paper's 1920x1080 device.

### Legacy generated-pattern acquisition (explicit opt-in)

Use `--generated-patterns` only when you intentionally want the old local
Zernike generator. Its basis support and normalization differ from the saved
NeuWS sequence. Omitting this option now selects replay; the script never
substitutes generated patterns when the replay files are missing.

Close SpinView and any HOLOEYE example that has exclusive control of a device,
then start with a short, low-risk run:

```powershell
python acquire_slm_camera.py `
  --generated-patterns `
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
  --generated-patterns `
  --num-patterns 100 `
  --frames-per-pattern 10 `
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

Each phase pattern stays on the SLM while the camera captures
`--frames-per-pattern` images (default **10**). The script settles once and
discards the configured buffer frames once after displaying the pattern, then
averages the retained images pixel by pixel. Averaging happens before scaling
or cropping. The mean uses float64 accumulation and is saved as float32 camera
counts, retaining fractional counts and avoiding integer overflow. It remains
on the same intensity scale as one exposure. `--exposure-ms` is the exposure
of each individual image.

By default, both the average and all contributing camera images are saved.
Use `--no-save-burst-frames` to keep only the averages and per-image metadata.
Use `--frames-per-pattern 1` for the earlier single-image behavior; in that
case the original integer camera image is retained without a duplicate burst.
The separate calibration utility continues to use its own `--repeats` setting.

The paper's 3.57 magnification is specific to its optical train. Perform the
aperture/fringe-shift calibration described on pages 4-5 for your setup. A
scalar `--magnification` applies only to aligned axes with equal sampling;
use the offline matrix option below for the measured 850 nm configuration.
Likewise, choose an exposure in
the paper's 90-120 ms range only after checking that your sample is not
saturating. Set `--wavelength-nm` to your actual laser wavelength because
HOLOEYE uses it for phase modulation; do not retain 532 nm if your laser is
different. Keep the beam blocked while testing software/device selection.

For the current 1920 x 1200 setup, the commands in `RUN_COMMANDS.txt` use
horizontal fringe lines and retain both zero and first orders in the same
camera frame. Use `--capture-fringe-first --no-neuws-processing` to capture
the pure steering fringe first, then add each random phase pattern to that
same fringe and capture another averaged measurement. The SLM and camera stay open
throughout the sequence. Each display update uses `--settle-ms` and
`--discard-frames`, including the initial fringe reference. Ensure the camera
ROI is set to the full sensor and both orders lie within its field of view.

The fringe reference uses the same averaging count. With the default of 10,
`--num-patterns 3 --capture-fringe-first` saves four averaged measurements
from 40 retained camera images before any steering retries (one fringe
reference and three random-plus-fringe measurements). Each adjustment or
recapture adds another averaged reference and its individual images.
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
captures the averaging burst, saves a new mean NPY/PNG and individual images,
and opens that preview for another decision.
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
omitting `--no-neuws-processing`. Full-frame averages and individual raw images
are still retained with the default settings.

The current run commands use `--frames-per-pattern 10 --save-burst-frames`,
`--output-format npy`, and omit `--save-full-patterns`. Averaged camera NPY/PNG
measurements, individual raw images, small phase-modulation arrays,
coefficients, and steering settings are retained.
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

### Current 850 nm export: scale and crop in camera orientation

Use the previously measured scale magnitudes, assigned to their dominant
camera directions: 6.810272393954523 camera pixels horizontally and
7.595985908414755 vertically per output pixel. The calibration lists these
in SLM-axis order (x=7.595985908414755, y=6.810272393954523); its matrix shows
that SLM x shifts mostly along camera y, and SLM y mostly along camera x.
Assigning these scale factors to camera axes does not transpose the images.

```powershell
python process_measurements.py data\measurement_850nm_001 --roi 200 200 900 1100 --preserve-camera-orientation --downsample-xy 6.810272393954523 7.595985908414755 --crop-size 128 --crop-center 658.9435440231936 752.8110040373784 --export-mat --output-dir processed\measurement_850nm_scaled_cropped_001
```

This bilinearly scales the full camera frame, then takes a fixed 128 x 128
crop around the recorded first-order center. No rotation, reflection, or
transpose is applied. The selected 900 x 1100 camera region supports about
132 x 145 pixels at these scales; a 128-square crop fits entirely within it.
A fully observed 256-square crop does not fit at the measured scale. Scaled
frame dimensions are rounded to integer sizes; the requested factors and the
resulting effective factors are both recorded under `scale_and_crop` in the
manifest. `--downsample-factor S` also supports a single common factor.

All measurements and the reference use the same transform. The processed raw
reference counts are `reference_processed.npy`, and the original full reference
is `reference/fringe_only.npy`. Camera count units are retained, with bilinear
values rounded to uint16; no intensity normalization is applied. The original
144 x 256 SLM phase arrays retain their orientation and values. All 100 pairs
and their source identities/timestamps are retained. Downstream reconstruction
must account for the 128-square camera crop and the measured camera-to-SLM
mapping explicitly; the phase/pupil grid is not cropped.

### Exact camera-crop diagnostic export

To export exact unscaled camera crops from the acquisition:

```powershell
python process_measurements.py data\measurement_850nm_001 --roi 200 200 900 1100 --preserve-camera-orientation --export-mat --output-dir processed\measurement_850nm_camera_axes_001
```

Without a downsampling option, `--preserve-camera-orientation` produces an
exact crop and rejects `--calibration-file`. Each image is
exactly `raw[200:1300, 200:1100]`: 1100 rows by 900 columns of original uint16
counts, with no rotation, reflection, transpose, filtering, or resizing. The
144 x 256 SLM modulation arrays are preserved in their original coordinates.
All 100 original pairs are retained here, including the five flagged source
indices 16, 41, 47, 66, and 91. Use source indices to select the same 95-frame
subset when comparing against the earlier fits.

New exports preserve the original acquisition manifest as `source_manifest.json`
and the original reference arrays under `reference/`. The processed manifest
also retains each frame's display, capture, and camera timestamps. Paths inside
`source_manifest.json` refer to the original acquisition; the processed pair
paths are in `manifest.json`. The source collection predates the later recaptures.

These camera-coordinate crops require a forward model with explicit sampling
into camera coordinates. They are not direct inputs to a model that assumes
256-square measurements in SLM coordinates. The physical camera-to-SLM mapping
must be represented before fitting; simply discarding the off-diagonal
calibration terms or stretching the crop into a square does not establish
correspondence.
The model-axis workflow below documents the earlier, reoriented export.

## 7. Measure SLM/camera Fourier sampling

`calibrate_slm_camera.py` implements the supplement's aperture/ramp calibration.
Replace the object with a small aperture, remove the aberrating sample, and
focus the aperture image. Keep the optical arrangement, camera ROI/binning,
and wavelength fixed between calibration and acquisition. Run the hardware
calibration command in `RUN_COMMANDS.txt` on the Windows acquisition PC.

The command starts with the 20-pixel horizontal fringe used for the uploaded
measurements. The current command reuses the 850 nm aperture ROI and enlarged
120-pixel centroid window. To select a different region interactively, omit
`--roi`: draw around only the steered aperture, excluding the static zero
order, then press Enter. Leave background margin at least as large as the
centroid radius around the spot.
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
registration. Apply it offline with the option described below.

If a spot leaves the search window, reduce `--step-cycles-per-pixel` or adjust
`--tracking-radius`, while keeping the static zero order outside the search.
Use `--analyze-only` to refit saved captures without opening hardware, optionally
with a corrected `--roi`. `--skip-review` saves an unapproved result for later
inspection. Each run/refit needs a new output directory.

## 8. Earlier model-axis export (changes image orientation)

This workflow does not preserve camera orientation. Supply an approved hardware calibration to resample the selected first order
with its full matrix, including rotation/reflection and unequal axis scales:

```powershell
python process_measurements.py data\measurement_850nm_001 --roi 200 200 900 1100 --calibration-file data\measurement_850nm_001\calibration_850nm.json --include-full-phase --export-mat --output-dir processed\measurement_850nm_originals_calibrated_001
```

Choose a new output directory on subsequent runs. This example selects the
left steered order in the uploaded 850 nm collection. Inspect `roi_preview.png`
and `reference_calibrated.png` to check the selection. The origin is the
calibration's predicted aperture position at the acquisition carrier; it stays
fixed across all patterns. `--camera-center X Y` can explicitly override that
origin. This is a choice of model-image origin, not an object-registration
measurement. The processor rejects incompatible wavelength, SLM/camera shapes,
carrier, phase-grid interpretation, or unapproved/synthetic calibrations.

The output retains `image_camera_crop` as exact uint16 camera values and the
original `phase_modulation_rad`. With calibration enabled, `image` is a
256 x 256 float32 array in the model's FFT axes. A Gaussian prefilter and
bilinear interpolation reduce aliasing; no background subtraction, per-frame
intensity normalization, or flux multiplier is applied. Filtering/interpolation
change individual values, so the unchanged crop remains available.

**Use `image_valid_mask` when fitting or comparing the calibrated image.**
Samples outside the selected camera ROI, sensor support, or filtering margin
are NaN, not observed zero intensity. Transparent PNG areas are missing data.
For this collection and crop, 17,900 of 65,536 model pixels (27.3%) are valid;
the central `[68:188, 68:188]` 120 x 120 image region is fully supported. A
complete observed 256 x 256 image does not fit at the measured sampling. Do not
stretch a smaller crop to 256 x 256, which would change the calibrated spacing.
When cropping the model image further, retain the 256-grid pupil/phase and
model the image crop explicitly.

Each pair also includes `phase_model_rad` and `pupil_mask`: the original
144 x 256 phase is embedded unchanged in rows 56:200 of a 256-square pupil,
with zero amplitude outside those rows. The carrier is absorbed into the camera
origin; it is not added to this modulation-only model phase. The optional
`phase_slm_rad` still represents the full displayed random-plus-carrier mask.
The pupil mask describes the generator rectangle, not a measured illumination footprint.
The camera ROI never crops or rotates the phase array. Calibrated MAT exports
contain `imsdata` plus `valid_mask`, and 256-square `proj_sim` plus `pupil_mask`.

```python
with np.load("processed/measurement_850nm_originals_calibrated_001/pairs/frame_0001.npz", allow_pickle=False) as pair:
    image = pair["image"]
    valid = pair["image_valid_mask"]
    phase = pair["phase_model_rad"]
    pupil = pair["pupil_mask"]
    # Compare a model prediction only on valid pixels: prediction[valid].
```

Source pattern indices, selected raw paths, source quality flags, the source
manifest hash, and the calibration hash/snapshot are recorded. Selected
recaptures are read from the manifest rather than inferred from filenames.
The supplied measurement ZIP predates the recaptures described in the session
notes: patterns 16, 41, 66, and 91 remain background-only, and 47 has weak
contrast. All are retained and flagged. The updated manifest and replacement
captures are needed to reproduce the preferred post-recapture collection.

## Output

Each run contains:

```text
sample_001/
  manifest.json
  raw/frame_0001.npy
  raw_bursts/frame_0001/sample_01.npy
  raw_bursts/frame_0001/sample_02.npy
  ...
  raw_bursts/frame_0001/sample_10.npy
  reference/fringe_only.npy
  raw_bursts/reference_0001/sample_01.npy
  ...
  previews/frame_0001.png
  previews/processed_0001.png
  processed/frame_0001.npy
  patterns/phase_modulation_0001.npy
  neuws_mat/SLM_raw1.mat
  neuws_mat/SLM_sim1.mat
```

`raw/frame_NNNN.npy` contains the full-sensor float32 average when averaging
is enabled. The contributing camera images retain their original dtype and
values under `raw_bursts/frame_NNNN/`. References use `reference/fringe_only.npy`
(with numbered suffixes for retries) and `raw_bursts/reference_NNNN/`.
With `--frames-per-pattern 1`, `raw/` contains the unchanged camera image.
`previews/` contains viewable 8-bit PNGs of the averages scaled between their 1st and 99th
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
frame identifiers, and display/capture timestamps. Each measurement's
`averaging` record contains the count, source/output dtypes, individual frame
IDs, camera timestamps, host capture times, and raw-image paths. Top-level
`frame_id` and `camera_timestamp_ns` identify the last contributing image;
`capture_start_time_ns` and `capture_end_time_ns` describe the burst interval.
`frames_completed` counts completed phase-pattern measurements, not exposures.
Preprocessing consumes the average and preserves this provenance in
`source_averaging` and `source_manifest.json`; individual-image paths refer to
the original acquisition directory.

A failed capture stops the run without publishing a partial average. After a
complete burst, the average is checkpointed before individual images and the
optional full SLM phase are written. Raw-image save progress is recorded in
the manifest; a storage failure leaves the average available and records
`raw_frames_save_error` when possible. A failed full-phase export leaves
`full_pattern_path` null and records `full_pattern_save_error` when possible.
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
- `--frames-per-pattern 10`: average ten camera images for each pattern and
  each fringe reference (default); use 1 for a single unchanged camera image.
- `--save-burst-frames`: retain individual camera images as well as averages
  (default). `--no-save-burst-frames` retains averages and metadata only.
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

Default replay preserves the supplied phase arrays, avoiding assumptions about
their Zernike sign, ordering, normalization, or corner support. The legacy
`--generated-patterns` mode records its convention and coefficients in the
manifest. Steering remains configured separately in radians per physical SLM
pixel (or with `--fringe-period-pixels`); use the current setup's wavelength,
SLM dimensions, and calibrated carrier as shown in the 633 nm command.

### Storage write errors

`OSError: ... requested and 0 written` at `np.save` means the output file
could not be written. Check available space on the acquisition PC:

```powershell
Get-PSDrive -PSProvider FileSystem
```

At 2000 x 1500 camera pixels, ten uint16 images use about 60 MB and their
float32 average uses another 12 MB per pattern. A 100-pattern run therefore
needs about **7.2 GB** for these arrays, plus references, previews, processing,
and optional MAT/phase exports. `--no-save-burst-frames` reduces this to about
1.2 GB of averages for 100 patterns.

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
