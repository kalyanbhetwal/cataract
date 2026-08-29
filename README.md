# HOLOEYE SLM + camera acquisition

`acquire_slm_camera.py` reproduces the acquisition recipe from pages 3-5 of
`sciadv.adg4671_sm.pdf`:

- HOLOEYE LETO-3 phase resolution: 1080 x 1920.
- Linear x fringe with slope `4*pi/3` radians per pixel.
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
python -m unittest -v test_acquire_slm_camera.py
```

The dry run generates realistic-sized simulated 16-bit frames and exercises
the same pattern, preprocessing, metadata, and file-writing path as hardware
mode. Keep the explicit test filename in the second command: broad unittest
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

## Output

Each run contains:

```text
sample_001/
  manifest.json
  raw/frame_0001.npy
  processed/frame_0001.npy
  patterns/phase_modulation_0001.npy
  neuws_mat/SLM_raw1.mat
  neuws_mat/SLM_sim1.mat
```

The full sensor frame is always retained in `raw/`. `processed/` contains the
3.57x downscaled, centered 256 x 256 image. The MAT files use the variable and
filename conventions expected by the public NeuWS reconstruction code:
`imsdata` in `SLM_rawN.mat` and `proj_sim` in `SLM_simN.mat`.

`manifest.json` records all settings, Zernike coefficients, file paths, camera
frame identifiers, and display/capture timestamps. It is rewritten after each
successful frame, so completed data remains described if acquisition stops.

Useful options:

- `--save-full-patterns`: retain every 1080 x 1920 displayed phase array.
- `--no-neuws-processing`: retain raw frames but skip scaling/cropping.
- `--output-format npy`: avoid the SciPy MAT-file dependency.
- `--discard-frames N`: discard buffered PySpin frames after each SLM update.
- `--slm-preselect STRING`: select an SLM in HEDS 4.x multi-SLM setups.
- `--heds-examples-dir PATH`: locate the installed SDK's `examples/HEDS` API.
- `--heds-api-version 4.1`: match an older installed HEDS SDK.
- `--playerone-sdk-dir PATH`: locate the Player One SDK root or `python` folder.
- `--camera-backend playerone`: switch from the default PySpin camera.
- `--wavelength-nm 532`: select the wavelength used by HOLOEYE's phase lookup.

The reference does not specify its Zernike sign/order convention, camera color
model, or the coordinate units attached to the stated fringe slope. This
implementation interprets the slope as radians per SLM pixel, records its
convention and coefficients in the manifest, and keeps those choices isolated
in `zernike_basis()`, `--fringe-slope-rad-per-pixel`, and `--pixel-format` so
they can be adjusted without changing the acquisition loop.
# cataract
