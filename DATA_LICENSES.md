# Dataset licenses and restrictions

This file is the English counterpart of the Stage 00 license record. Raw data and
model weights are not redistributed by this repository.

## LOL

- Hugging Face ID: `geekyrakshit/LoL-Dataset`
- Pinned revision: `bdef22d0f5803da8775114496a6689333265e157`
- Archive size: 347,171,015 bytes
- Archive SHA256: `419fddc48958cd0f5599939ee0248852a37ceb8bb738c9b9525e95b25a89de9a`
- License: **unknown**. The Hugging Face repository does not declare a clear
  license.
- Restrictions:
  - do not publish or redistribute the raw data,
  - this repository stores only the download script and metadata,
  - cite the LOL paper when publishing results,
  - ask the course instructor before any redistribution.
- Paper: Chen et al., "Deep Retinex Decomposition for Low-Light Enhancement"
  (BMVC 2018).

## LOL-v2 Real

- Hugging Face ID: `okhater/lolv2-real`
- Pinned revision: `db2979107f7c22d372c6e4f1b06fd6203069af0a`
- License: **Apache-2.0**, confirmed in the dataset card and `LICENSE` file.
- The data is downloaded by script and is not committed to git.
- Paper: Yang et al., "Sparse Gradient Regularized Deep Retinex Network for
  Robust Low-Light Enhancement" (IEEE TIP 2021).

## LOL-v2 Synthetic

- Hugging Face ID: `okhater/lolv2-synthetic`
- Pinned revision: `2fe16f9ebde66b20df7222999f53843f68c61aa4`
- License: **Apache-2.0**.
- Status: **backup only**. It is not downloaded by default and is not used in
  training, validation, test or ablations. Any later use requires an explicit
  user decision and a manifest update.

## Repository policy

- `data/raw/` is ignored by git.
- `data/manifests/` and `data/splits/` contain metadata and hashes only.
- Checkpoints, predictions and demo uploads are not committed.
- Do not publish data or trained weights without checking the license and
  obtaining explicit permission.

## Environment and dependency licenses

Package approvals and installed versions are recorded in
`Outdated/Plans/PACKAGE_APPROVALS.md` and `requirements.lock.txt`. `pyiqa` is
used only for NIQE evaluation; its upstream NTU S-Lab terms apply.
