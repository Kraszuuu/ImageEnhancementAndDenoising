# Low-Light Image Enhancement and Denoising

Comparison of classical and CNN methods for low-light image enhancement and
denoising on paired LOL and LOL-v2 Real data. The project covers data auditing,
frozen metrics, classical baselines, compact U-Net models, combined pipelines,
pre-registered ablations and an interactive demo.

Raw datasets, model weights and intermediate predictions are not committed.
Datasets are downloaded from pinned Hugging Face revisions; weights are
reproduced by training from configs and seeds.

## Requirements

- Windows 11 x86_64 (verified on build 26200, Windows 11 25H2).
- Python 3.14.
- NVIDIA RTX 3060 12 GB with driver 616.56 (CUDA UMD 13.4) for the reported
  GPU timings. CPU inference works but is slower.
- PowerShell 5.1 or later.

## Python 3.14 setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt `
  --extra-index-url https://download.pytorch.org/whl/cu130
```

`requirements.in` lists direct dependencies. `requirements.lock.txt` records the
exact environment used for the reported results, including
`torch==2.13.0+cu130` and `torchvision==0.28.0+cu130`. Streamlit is optional for
non-demo work and is included in the lock file.

## Data download

The downloader uses pinned revisions by default. Requires network access.

```powershell
python scripts\downloadData.py --dataset lol
python scripts\downloadData.py --dataset lolv2-real
python scripts\downloadData.py --dataset lolv2-synthetic --backup-only
python scripts\downloadData.py --check
```

Pinned sources:

| Dataset | Hugging Face ID | Revision | License | Use |
|---|---|---|---|---|
| LOL | `geekyrakshit/LoL-Dataset` | `bdef22d0f5803da8775114496a6689333265e157` | unknown | primary |
| LOL-v2 Real | `okhater/lolv2-real` | `db2979107f7c22d372c6e4f1b06fd6203069af0a` | Apache-2.0 | primary |
| LOL-v2 Synthetic | `okhater/lolv2-synthetic` | `2fe16f9ebde66b20df7222999f53843f68c61aa4` | Apache-2.0 | backup only |

LOL has no clear license. Do not redistribute its raw images. See
`DATA_LICENSES.md` for restrictions. `data/raw/` is ignored by git.

## Manifests and frozen splits

```powershell
python scripts\auditData.py
python scripts\buildManifests.py
python scripts\buildSplits.py
```

This writes pair manifests, scene-grouped train/validation/test splits and their
checksums under `data/manifests/` and `data/splits/`. The split seed is `0`;
scene overlap and file hashes are in `data/splits/split_manifest.json`.

## Tests and lint

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

`mypy src` is configured but is not a required gate for this project.

## Training

The full pipeline can be orchestrated with:

```powershell
python scripts\reproduceAll.py --stage all --dry-run
python scripts\reproduceAll.py --stage all
```

`--stage` accepts `data`, `metrics`, `denoising`, `brightening`, `combined`,
`analysis` or `all`. Existing checkpoints and locked test outputs are reused by
default. Use `--force` to rerun training and non-locked evaluations. Locked test
outputs are not overwritten by `reproduceAll.py`. Dataset downloads require
`--allow-download`. The script writes `results/final/run_metadata.json` and does
not commit anything.

Approximate full retraining time on the verified RTX 3060 12 GB, excluding data
download: **14-20 hours** including tuning, evaluation and ablations. A single
one- or two-image smoke evaluation takes seconds to a few minutes depending on
the method.

Manual stage commands:

```powershell
# Metrics and controlled noise
python scripts\validateMetrics.py
python scripts\calibrateNoise.py

# Classical and CNN denoising
python scripts\tuneClassicalDenoising.py
python scripts\evaluateClassicalDenoising.py
python scripts\benchmarkUnet.py
python scripts\trainDenoising.py --config configs\training\denoising.yaml --seed 0
python scripts\trainDenoising.py --config configs\training\denoising.yaml --seed 1
python scripts\trainDenoising.py --config configs\training\denoising.yaml --seed 2
python scripts\evaluateDenoising.py
python scripts\evaluateDFixed.py

# Classical and CNN brightening
python scripts\tuneClassicalBrightening.py
python scripts\evaluateClassicalBrightening.py
python scripts\precomputeDFixed.py
python scripts\trainBrightening.py --track raw --seed 0
python scripts\trainBrightening.py --track raw --seed 1
python scripts\trainBrightening.py --track raw --seed 2
python scripts\trainBrightening.py --track d_fixed --seed 0
python scripts\trainBrightening.py --track d_fixed --seed 1
python scripts\trainBrightening.py --track d_fixed --seed 2
python scripts\evaluateBrightening.py

# Combined pipelines and end-to-end models
python scripts\precomputePipelineSources.py --checkpoint checkpoints\denoising_cnn\seed0\best.pt --cache-dir outputs\pipeline_source_cache\cnn\seed0
python scripts\precomputePipelineSources.py --checkpoint checkpoints\denoising_cnn\seed1\best.pt --cache-dir outputs\pipeline_source_cache\cnn\seed1
python scripts\precomputePipelineSources.py --checkpoint checkpoints\denoising_cnn\seed2\best.pt --cache-dir outputs\pipeline_source_cache\cnn\seed2
python scripts\trainPipelineBrightener.py --denoiser classical --seed 0
python scripts\trainPipelineBrightener.py --denoiser classical --seed 1
python scripts\trainPipelineBrightener.py --denoiser classical --seed 2
python scripts\trainPipelineBrightener.py --denoiser cnn --seed 0
python scripts\trainPipelineBrightener.py --denoiser cnn --seed 1
python scripts\trainPipelineBrightener.py --denoiser cnn --seed 2
python scripts\trainEndToEnd.py --input-mode real --seed 0
python scripts\trainEndToEnd.py --input-mode real --seed 1
python scripts\trainEndToEnd.py --input-mode real --seed 2
python scripts\trainEndToEnd.py --input-mode controlled --seed 0
python scripts\trainEndToEnd.py --input-mode controlled --seed 1
python scripts\trainEndToEnd.py --input-mode controlled --seed 2
python scripts\evaluatePipelines.py
```

Every run writes a `run_metadata.json` with git state, environment, config hash,
split hashes, seed and command. Checkpoints stay in ignored `checkpoints/`.

## Evaluation

```powershell
python scripts\evaluate.py --stage all --dry-run
python scripts\evaluate.py --stage pipelines
python scripts\evaluate.py --stage ablations
```

`scripts/evaluate.py` dispatches the frozen evaluator for a stage and forwards
extra arguments. The underlying scripts refuse to overwrite locked test outputs
unless a documented `--force` is supplied. The main test datasets are
`data/splits/lol_test.csv` (`n=15`) and `data/splits/lolv2_test.csv` (`n=100`).

The analysis and final tables can be regenerated with:

```powershell
python scripts\analyzeResults.py --config configs\ablations\ablations.yaml
python scripts\makeFigures.py
python scripts\makeFailurePanels.py
python scripts\reproduceAll.py --stage analysis --skip-tests
```

## Demo

```powershell
.\scripts\runDemo.ps1
```

The Streamlit app is implemented in `src/lowlight/demo/app.py` with shared
validation and inference in `src/lowlight/demo/inference.py`. It accepts PNG or
JPEG uploads, supports RGB and grayscale inputs, limits images to 2048 px per
side and 4,000,000 pixels, reports inference time and optional NIQE, and shows
side-by-side output. Models are loaded from frozen configs and cached per
method/seed. A missing checkpoint produces a clear error; retrain from this
README or select another method. The demo does not train, download data or
modify experiment configs.

The demo is not a benchmark. Uploaded images have no reference target, so PSNR,
SSIM and LPIPS cannot be computed. Use `results/final/` and the handouts for
benchmark conclusions.

## Expected results

All numbers are means over three seeds for learned methods. PSNR in dB.
Unless stated otherwise, the primary dataset is LOL-v2 Real with `n=100`.

| Experiment | Condition | Result |
|---|---|---|
| Denoising BM3D vs U-Net | low | BM3D `39.529` vs U-Net `36.663` |
| Denoising BM3D vs U-Net | medium | BM3D `34.910` vs U-Net `33.822` |
| Denoising BM3D vs U-Net | high | BM3D `28.516` vs U-Net `31.509` |
| Brightening gamma vs U-Net | raw track | gamma `17.565` vs U-Net `22.874` |
| Brightening gamma vs U-Net | D_fixed track | gamma `17.920` vs U-Net `21.604` |
| Pipeline order | real input | D-to-B `17.920` vs B-to-D `18.595` |
| Pipeline order | high noise | D-to-B `13.489` vs B-to-D `10.217` |
| End-to-end vs 2x2 pipeline | real input | E2E `22.874` vs `cnn_cnn` `22.217` |
| Loss function | real input | L1 `22.455` vs L1+LPIPS `22.874` |
| Dataset composition | LOL-v2 Real test | mixed `22.874`, LOL-only `22.299`, LOL-v2-only `19.946` |
| Synthetic brightening | LOL-v2 Real test | `30.221` |
| Synthetic denoising | LOL-v2 Real low/medium/high | `28.169` / `27.705` / `27.461` |

Key conclusions:

- BM3D is competitive at low and medium noise; the U-Net wins at high noise by
  `+2.993 dB`.
- CNN brightening beats gamma on both brightening tracks and all four metrics.
- Denoise-then-brighten is better on real input; brighten-then-denoise collapses
  at high noise because brightening amplifies noise before denoising.
- End-to-end real is the strongest learned method on real input, but `cnn_cnn`
  remains a strong modular alternative.
- Perceptual loss improves PSNR, SSIM and LPIPS but costs about `25.1` instead of
  `16.1` minutes per seed.
- Mixing LOL and LOL-v2 helps on LOL-v2, but LOL-v2-only is unstable in-domain.
- Synthetic denoising seed 2 failed to converge and is retained in all reports.

## Limitations

- LOL test has only 15 images; main statistical conclusions use LOL-v2 Real.
- The denoising target `L + n -> L` still contains real sensor noise.
- Brightening amplifies noise; the `D_fixed` track removes detail, so the raw
  track is reported separately.
- NIQE is no-reference and should not be compared across datasets without care.
- Synthetic control is an easier mechanism check, not a substitute for real
  paired low-light data.
- LOL has no clear license; raw data and weights must not be published without
  permission.
- Formal Stage 00B package-approval evidence remains a project-level caveat.
- The demo is optional and is not part of benchmark claims.

## Reproducibility notes

`scripts/reproduceAll.py` verifies data hashes, rebuilds manifests and splits,
runs tests, optionally trains with seeds, evaluates, analyzes results and writes
`results/final/run_metadata.json`. It records git state, Python and package
versions, CUDA/driver information, config and split hashes, seeds and commands.

Suggested reproduction tolerance after retraining is PSNR `+/- 0.1 dB` and SSIM
`+/- 0.001`. Exact bitwise reproduction is not guaranteed because CUDA,
DataLoader workers and package versions can introduce nondeterminism. If a
retrained result falls outside tolerance, inspect CUDA nondeterminism,
DataLoader behavior, package versions and checkpoint selection.

Locked test evidence:

- `results/ablations/evaluation/loss_l1/test_lock.json`
- `results/ablations/evaluation/composition/test_lock.json`
- `results/ablations/evaluation/synthetic_brightening/test_lock.json`
- `results/ablations/evaluation/synthetic_denoising/test_lock.json`
- `results/pipelines/evaluation/test_lock.json`
- `results/brightening_cnn/evaluation/test_lock.json`

Do not rerun locked tests after seeing their results. Negative results are kept.

## Repository layout

```text
configs/        frozen experiment configuration
data/raw/       downloaded datasets (ignored)
data/manifests/ manifests and checksums (committed)
data/splits/    frozen train/validation/test splits (committed)
src/lowlight/   library code, including the optional demo
scripts/        CLI entry points and reproduction scripts
tests/          unit and integration tests
results/        tables, figures and locked test evidence (committed)
results/final/  consolidated Stage 08 tables and figures
checkpoints/    model checkpoints (ignored)
outputs/        intermediate predictions and caches (ignored)
```

## Status and documentation

Stages 00-08 are complete. Stage 02 results are in
`results/classical_denoising/`; Stage 03 in `results/denoising_cnn/`; Stage 04 in
`results/classical_brightening/`; Stage 05 in `results/brightening_cnn/`;
Stage 06 in `results/pipelines/`; Stages 07-08 in `results/ablations/` and
`results/final/`.

Handouts and plans are under `Outdated/Plans/`, including
`Outdated/Plans/handouts/final/HANDOUT.md` and
`Outdated/Plans/handouts/stage08/HANDOUT.md`. That directory is ignored by git
and must be force-added if it should be versioned.

## Commit policy

- No raw datasets, checkpoints, predictions or uploaded images in git.
- Results and metadata are committed only after a frozen evaluation.
- The user decides whether and when to commit, push, tag or publish.
- Suggested final commit message: `stage08: add demo and reproducibility package`.
