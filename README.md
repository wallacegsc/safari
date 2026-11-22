# MLPerf Tiny Visual Wake Words on SA-FARI

This repository holds the code, frame lists and results for an evaluation of the MLPerf Tiny Visual Wake Words (VWW) model on SA-FARI camera-trap footage. It covers three training runs:

| Run | Input | Initialization |
|---|---|---|
| `scratch96_s0` | 96×96 | random (from scratch) |
| `ft96_s0` | 96×96 | MLPerf Tiny `vww_96.h5` |
| `ft128_s0` | 128×128 | MLPerf Tiny `vww_96.h5` |

All stages use fixed seeds (seed 0).

## What is included and what is not

Included:
- `code/`: the Python modules, the three notebooks (with their outputs) and the figures (`code/display/`).
- `code/third_party/mlperf_tiny/`: [MLPerf Tiny](https://github.com/mlcommons/tiny) as a git submodule, pinned at commit `4addd0fa08d216e20637637874e084895f289da4`.
- `data/plans/`: the exact frame lists, labels and train/val/test splits.
- `logs/<run>/`: configuration, per-epoch metrics, per-frame predictions, evaluation results and console output of the three runs.
- `checkpoints/<run>/`: `best.h5`, `last.h5`, `last_float32.tflite` and `last_int8.tflite`.

Not included: the SA-FARI annotations and frames, and the resized caches made from them. Instructions for getting them follow.

## Setup

```bash
git clone --recurse-submodules <this repository> safari
cd safari
python3.12 -m venv .venv && source .venv/bin/activate   # the runs used Python 3.12.15
pip install -r requirements_used.txt
```

If you cloned without `--recurse-submodules`, run `git submodule update --init`.

The runs used TensorFlow 2.21.0 (CUDA 12 wheels from `requirements_used.txt`) on one NVIDIA RTX 3050 Laptop GPU with 4 GB, under WSL2. Results are deterministic on the same hardware and software (`enable_op_determinism`). A different GPU or library version can give slightly different numbers.

**Clone the repository to `~/safari`.** A few cells in `code/display.ipynb` write figures to `~/safari/...`. If you clone elsewhere, edit those paths first.

## Data

### 1. Annotations (Hugging Face, gated)

The annotations come from [`facebook/SA-FARI`](https://huggingface.co/datasets/facebook/SA-FARI), revision `b38cec702475f707faa759ac49d5b2082467c6e1`. Accept the dataset terms on Hugging Face, log in (for example `hf auth login`, or set `HF_TOKEN`), and download the two files into `data/hf_sa_fari/annotation/`:

```python
from huggingface_hub import snapshot_download
snapshot_download("facebook/SA-FARI", repo_type="dataset",
                  revision="b38cec702475f707faa759ac49d5b2082467c6e1",
                  allow_patterns=["annotation/sa_fari_train.json", "annotation/sa_fari_test.json"],
                  local_dir="data/hf_sa_fari")
```

Expected result:

```
data/hf_sa_fari/annotation/sa_fari_train.json   (~834 MB)
data/hf_sa_fari/annotation/sa_fari_test.json    (~49 MB)
```

### 2. Frames (public bucket, downloaded by the notebook)

The frames come from the public SA-FARI Google Cloud Storage bucket `cxl-public-camera-trap`, with anonymous access and no credentials. `code/pipeline.ipynb` downloads only the 10% sample listed in `data/plans/plan_*_download_plan.json`, about 6.8 GB in total. They go to:

```
data/google_sa_fari/sa_fari/sa_fari_train/JPEGImages_6fps/<video>/<frame>.jpg
data/google_sa_fari/sa_fari/sa_fari_test/JPEGImages_6fps/<video>/<frame>.jpg
```

The resized caches used for training go to `data/vww96/sa_fari_{train,test}/` and `data/vww128/sa_fari_{train,test}/`.

## Run order

The notebooks expect Jupyter to be started in `code/`, and use paths like `../data`. Training runs from a terminal in the repository root.

1. **`code/pipeline.ipynb`** (Jupyter started in `code/`: `cd code && jupyter lab`):
   - Cell 1: 10% sampling and frame download → `data/plans/plan_*`, `manifest_*_download_plan.csv`, and the frames.
   - Cell 2: labeling (largest animal > 2.5% of the frame; frames with only smaller animals are dropped; two camera-setup videos excluded) → `data/plans/manifest_vww_{train,test}.csv`.
   - Cell 3: train/val split by camera location (15% val, seed 0) → `data/plans/split_{train,val}.csv`.
   - Cell 4: 96×96 and 128×128 caches → `data/vww96/`, `data/vww128/`.
   - Skip cells 5–7. Cells 5–6 recorded the environment, and cell 6 would overwrite `requirements_used.txt`. Cell 7 holds the training commands but only works from the repository root (next step).
2. **Training**, from a terminal in the repository root with the virtual environment active. These are the commands as they were run:

   ```bash
   H5=code/third_party/mlperf_tiny/benchmark/training/visual_wake_words/trained_models/vww_96.h5
   python3 code/train_vww_safari.py --run-name scratch96_s0 --seed 0
   python3 code/train_vww_safari.py --run-name ft96_s0      --seed 0 --init $H5
   python3 code/train_vww_safari.py --run-name ft128_s0     --seed 0 --init $H5 --image-size 128
   ```

   Each run writes `logs/<run>/` and `checkpoints/<run>/`. A run refuses to start if its folders already exist, so move the included results aside before retraining. Training took 106 min, 106 min and 162 min on the GPU above.
3. **`code/pipeline.ipynb` cell 9**: float32 and int8 TFLite conversion → `checkpoints/<run>/last_{float32,int8}.tflite`.
4. **`code/locations.ipynb`**: camera-location statistics and the location figures (`code/display/locations/`).
5. **`code/display.ipynb`**: dataset statistics, training curves, test metrics and confusion matrices, camera-level bootstrap confidence intervals (10,000 resamples, seed 0), recall by animal size, the TFLite evaluation, and the remaining figures (`code/display/`).

You don't need to retrain to check the results. The `display.ipynb` cells that read only `logs/`, such as training curves, test metrics, bootstrap and recall by size, run on the included results. The other cells also need the annotations, frames or caches.

## License

- **Code** in this repository: Apache License 2.0 (see [`LICENSE`](LICENSE)).
- **MLPerf Tiny** (`code/third_party/mlperf_tiny`, including `vww_96.h5`): Apache License 2.0, © MLCommons, under its own license.
- **SA-FARI data**: covered by the SA-FARI license (the Hugging Face dataset card lists CC BY-NC 4.0). This applies to the annotations and frames, the frame lists and labels derived from them (`data/plans/`, per-frame results in `logs/`), and the SA-FARI frames shown in the figures (`code/display/`). Frames in the figures are resized or cropped, and some have bounding boxes drawn on them. If you use the data, cite the SA-FARI paper:

  > Wasmuht et al. *The SA-FARI Dataset: Segment Anything in Footage of Animals for Recognition and Identification.* CVPR 2026. arXiv:2511.15622.
