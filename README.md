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
- Camera defaults of 100 ms exposure, gain 0, gamma 1, and unpacked `Mono16`.
- Paper-compatible downscaling by 3.57 followed by a centered 256 x 256 crop.

The script supports HOLOEYE Display SDK 4.x (`HEDS`) and the older
`holoeye.slmdisplaysdk` interface. Camera control uses FLIR/Teledyne Spinnaker
through `PySpin`.

## 1. Validate without hardware

Use the Python environment in which NumPy is installed:

```powershell
python acquire_slm_camera.py --dry-run --num-patterns 3 --output-format npy
python -m unittest -v
```

The dry run generates realistic-sized simulated 16-bit frames and exercises
the same pattern, preprocessing, metadata, and file-writing path as hardware
mode.

## 2. Install vendor software on the acquisition PC

1. Install the HOLOEYE SLM Display SDK supplied through the HOLOEYE customer
   site. Add its `api/python` and `examples` directories to `PYTHONPATH` if its
   installer does not do so. The script defaults to HEDS API 4.2; pass
   `--heds-api-version 4.1` or `4.0` if that is the installed SDK.
2. Install a Spinnaker SDK version that supports the connected Grasshopper3,
   including its matching `PySpin` Python package.
3. Install this script's normal Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

HOLOEYE currently distributes its desktop Display SDK for Windows. Run
hardware acquisition on the Windows computer driving the SLM, not on this Mac.

## 3. Acquire data

Close SpinView and any HOLOEYE example that has exclusive control of a device,
then start with a short, low-risk run:

```powershell
python acquire_slm_camera.py `
  --num-patterns 3 `
  --camera-serial YOUR_CAMERA_SERIAL `
  --output-dir data/test_run
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
  --camera-serial YOUR_CAMERA_SERIAL `
  --output-dir data/sample_001
```

The paper's 3.57 magnification is specific to its optical train. Perform the
aperture/fringe-shift calibration described on pages 4-5 for your setup and
pass the measured value as `--magnification`. Likewise, choose an exposure in
the paper's 90-120 ms range only after checking that your sample is not
saturating. Keep the beam blocked while testing software/device selection.

For a color Grasshopper3, select an unpacked 16-bit Bayer format supported by
the camera, such as `BayerRG16`, instead of `Mono16`. Available formats can be
checked in SpinView. Do not use a packed 12-bit format because `GetNDArray()`
cannot expose all packed formats directly.

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
- `--discard-frames N`: discard buffered frames after each SLM update.
- `--slm-preselect STRING`: select an SLM in HEDS 4.x multi-SLM setups.
- `--heds-api-version 4.1`: match an older installed HEDS SDK.

The reference does not specify its Zernike sign/order convention, camera color
model, or the coordinate units attached to the stated fringe slope. This
implementation interprets the slope as radians per SLM pixel, records its
convention and coefficients in the manifest, and keeps those choices isolated
in `zernike_basis()`, `--fringe-slope-rad-per-pixel`, and `--pixel-format` so
they can be adjusted without changing the acquisition loop.
# cataract
