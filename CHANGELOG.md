# Changelog

## 1.1.0

Reorganises the code into an installable `ssai3d` package with a command line
interface. The code as published with the paper is tagged `v1.0-paper`.

On the demo stack, the semi-synthetic training pairs, the axial input slices
and the surgeon's layer ranking are bit-identical to v1.0.

### Added
- `ssai3d run <config.yml>`: data preparation, layer ranking, fine-tuning,
  restoration and fusion from one config. Finished steps are skipped on rerun;
  run time and peak GPU memory per step go to `metrics.json`.
- `ssai3d restore`: apply an existing network to a stack.
- `ssai3d download`: fetch the released weights and the demo stack, with checksums.
- `gpu` / `max_mem_gb` options: pin to one GPU and cap its memory.
- Tiled inference with blended overlaps (`restore.tile`), used automatically when
  a slice runs out of memory. BasicSR's `grids()` could not run at inference
  (it reads `self.gt`).
- `pyproject.toml`, `environment.yml` (PyTorch 2.5, CUDA 12.4) and unit tests.

### Changed
- SSAI-3D code moved from `basicsr/{data,trainer,model_train,demo_test}` to
  `ssai3d/`. `basicsr/` now only holds the vendored NAFNet/BasicSR engine.
- The restored volume is saved as one `(Z, Y, X)` TIFF. Per-slice outputs are
  single-channel float32 (the channel mean the original fusion used).
- Options are passed as arguments; nothing is read from `sys.argv`, and paths no
  longer depend on the working directory.
- Training resumes only when asked (`resume=True`). Logs and TensorBoard files
  go to the experiment folder.
- Notebooks rewritten around the new API.

### Fixed
- `generate_raw_data` always cropped the stack to `[:80, :400, :500]`. The crop
  is now an explicit `crop=` argument, off by default.
- `test_xy` held axial slices instead of lateral ones (affects the denoising path).
- Max-projection pairs always used a downsampling rate of 5, ignoring `downsample_rate`.
- `generate_zs_dataset` wrote to `./demo_dataset` whatever its input was, took the
  bottom-right corner instead of a centre crop, and was not seeded.
- The surgeon MLP loaded `./experiments/sur.pt` relative to the working directory
  and fell back to random weights silently, including on PyTorch < 1.13
  (`weights_only`). It now ships with the package and fails loudly.
- Inference ran on every visible GPU (`DataParallel`).
- `scipy.signal.gaussian` was removed in SciPy 1.13.
- A validation set was built from the home directory (`dataroot: ~/`).
- No training log file was written when another model had been loaded earlier
  in the same process (as in the notebooks).

### Removed
- NAFNet option files for GoPro/SIDD/REDS/NAFSSR, `demo/tmp`, empty modules,
  committed `__pycache__` and `.DS_Store` files.
