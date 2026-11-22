"""
Generic training run for the MLPerf Tiny VWW model on SA-FARI.

Reused unchanged from mlcommons/tiny (benchmark/training/visual_wake_words):
  vww_model.mobilenet_v1, train_vww.train_epochs, train_vww.IMAGE_SIZE /
  BATCH_SIZE, the exact ImageDataGenerator settings, and the optional
  "start from a saved model" path of train_vww.main (load_model(argv[1])).

With no arguments it reproduces the baseline run exactly (MLPerf schedule
20@1e-3, 10@5e-4, 20@2.5e-4, from scratch, our split CSVs and 96x96 cache).

Resolution: --image-size N (default 96 = MLPerf). The model is always MLPerf's
mobilenet_v1, built from their own vww_model.py source with only the input
size changed (their file is not edited). At 96 their function is called as is.
--init weights (e.g. vww_96.h5) load into any resolution: conv/BN/dense
weights do not depend on the image size. Images come from data/vww<N>/.

Model selection (does not change training): after every epoch the model is
scored on the CLEAN val set (no augmentation, same preprocessing as
SaFariVWW.diagnose). The epoch with the best --select metric is "best"; the
final epoch is "last" (= the MLPerf protocol). Both are evaluated at the end.
Test is never looked at during training.

Everything that can change between runs is an argument; nothing run-specific
is hard-coded. One execution = one training run. Each run writes:
  <logs-dir>/<run-name>/config.json          resolved settings (reproducibility)
  <logs-dir>/<run-name>/console.log          full console output
  <logs-dir>/<run-name>/epochs.csv           one row per epoch: stage, lr, Keras train/val
                                             metrics, clean-val metrics (cv_*), is_best, time
  <logs-dir>/<run-name>/val_preds/epNNN.npy  clean-val P(animal) per frame, val.csv order
  <logs-dir>/<run-name>/val_preds/labels.npy clean-val labels, same order
  <logs-dir>/<run-name>/data/*.csv           the exact train/val/test lists used
  <logs-dir>/<run-name>/diag_<model>_<set>.json  SaFariVWW.diagnose (+ _frames.csv)
  <logs-dir>/<run-name>/summary.json         final metrics (best and last) and paths
  <ckpt-dir>/<run-name>/epochs/epNNN.h5      model after every epoch (no optimizer state)
  <ckpt-dir>/<run-name>/best.h5              best clean-val epoch (by --select)
  <ckpt-dir>/<run-name>/last.h5              final epoch
  <ckpt-dir>/<run-name>/stage<k>.h5          model after each stage (if --save-stages)

To pick an epoch by another metric later: compute it from val_preds/epNNN.npy
and labels.npy, then load ckpt/<run>/epochs/epNNN.h5. Choose on val only.

Examples (venv active, from ~/safari):
  python3 code/train_vww_safari.py --run-name baseline
  python3 code/train_vww_safari.py --run-name ft_coco \
      --init code/third_party/mlperf_tiny/benchmark/training/visual_wake_words/trained_models/vww_96.h5
  python3 code/train_vww_safari.py --run-name smoke --smoke
  python3 code/train_vww_safari.py --run-name ft_128 --image-size 128 --init <vww_96.h5>
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")      # tf.keras = Keras 2, as MLPerf
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

ROOT = Path(__file__).resolve().parent.parent           # ~/safari (defaults only)
CLASSES = ["0", "1"]          # index 0 = empty, 1 = animal (as non_person/person in MLPerf)
MLPERF_SCHEDULE = "20:0.001,10:0.0005,20:0.00025"       # train_vww.main


# ---------------------------------------------------------------- arguments

def parse_args(argv=None):
    d = ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None,
                    help="name of this run (default: vww_<timestamp>)")
    # data
    ap.add_argument("--train-csv", default=str(d / "data/plans/split_train.csv"))
    ap.add_argument("--val-csv", default=str(d / "data/plans/split_val.csv"))
    ap.add_argument("--test-csv", default=str(d / "data/plans/manifest_vww_test.csv"),
                    help="'' to skip test evaluation")
    ap.add_argument("--image-size", type=int, default=96,
                    help="input resolution N (NxN); 96 = MLPerf. Multiple of 32.")
    ap.add_argument("--train-images", default=None,
                    help="cache root for train and val paths (default data/vww<N>/sa_fari_train)")
    ap.add_argument("--test-images", default=None,
                    help="default data/vww<N>/sa_fari_test")
    ap.add_argument("--ann-train", default=str(d / "data/hf_sa_fari/annotation/sa_fari_train.json"),
                    help="annotation for val diagnosis ('' to skip diagnosis)")
    ap.add_argument("--ann-test", default=str(d / "data/hf_sa_fari/annotation/sa_fari_test.json"))
    # model / schedule
    ap.add_argument("--mlperf-dir",
                    default=str(d / "code/third_party/mlperf_tiny/benchmark/training/visual_wake_words"))
    ap.add_argument("--init", default=None,
                    help="start from this saved model (.h5) instead of a fresh mobilenet_v1")
    ap.add_argument("--schedule", default=MLPERF_SCHEDULE,
                    help="stages 'epochs:lr,...' passed to train_vww.train_epochs (default MLPerf)")
    ap.add_argument("--seed", type=int, default=0,
                    help="one seed for everything: weight init, subsets, shuffle, augmentation")
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                    help="allow non-deterministic GPU kernels (faster, results vary run to run)")
    # run control
    ap.add_argument("--smoke", action="store_true",
                    help="quick pipeline check: schedule 1:0.001 and small subsets")
    ap.add_argument("--limit-train", type=int, default=None, help="random subset of train rows")
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--select", default="balanced_accuracy",
                    choices=["balanced_accuracy", "auc", "loss", "f1_animal", "accuracy"],
                    help="clean-val metric that defines best.h5 (loss: lower is better)")
    ap.add_argument("--no-epoch-checkpoints", dest="epoch_checkpoints", action="store_false",
                    help="do not keep a model per epoch (best.h5/last.h5 are still saved)")
    ap.add_argument("--save-stages", action="store_true",
                    help="also save the model after each schedule stage")
    ap.add_argument("--fit-verbose", type=int, default=None, choices=[1, 2],
                    help="override Keras verbosity (2 = one line per epoch); default = MLPerf's")
    # outputs
    ap.add_argument("--logs-dir", default=str(d / "logs"))
    ap.add_argument("--ckpt-dir", default=str(d / "checkpoints"))
    a = ap.parse_args(argv)

    if a.smoke:
        a.schedule = "1:0.001" if a.schedule == MLPERF_SCHEDULE else a.schedule
        a.limit_train = a.limit_train or 2000
        a.limit_val = a.limit_val or 500
        a.limit_test = a.limit_test or 500
    if a.image_size < 32 or a.image_size % 32:
        raise SystemExit(f"--image-size must be a multiple of 32 (got {a.image_size})")
    a.train_images = a.train_images or str(d / f"data/vww{a.image_size}/sa_fari_train")
    a.test_images = a.test_images or str(d / f"data/vww{a.image_size}/sa_fari_test")
    a.run_name = a.run_name or "vww_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    a.stages = parse_schedule(a.schedule)
    return a


def parse_schedule(text):
    stages = []
    for part in text.split(","):
        ep, lr = part.split(":")
        ep, lr = int(ep), float(lr)
        if ep <= 0 or lr <= 0:
            raise ValueError(f"bad schedule stage {part!r}")
        stages.append((ep, lr))
    if not stages:
        raise ValueError("empty schedule")
    return stages


# ---------------------------------------------------------------- logging

class Tee:
    """Duplicate a stream into a log file."""
    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", buffering=1)
    def write(self, s):
        self.stream.write(s); self.file.write(s)
    def flush(self):
        self.stream.flush(); self.file.flush()
    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()
    def fileno(self):
        return self.stream.fileno()


CV_FIELDS = ["cv_loss", "cv_accuracy", "cv_balanced_accuracy", "cv_auc",
             "cv_recall_animal", "cv_recall_empty", "cv_precision_animal", "cv_f1_animal"]


def make_epoch_logger(tf, path, state, epoch_hook=None):
    """Keras callback writing one row per epoch with run-level context.
    epoch_hook(model, global_epoch) -> dict of extra columns (clean-val metrics)."""
    fields = (["global_epoch", "stage", "stage_epoch", "lr", "loss", "accuracy",
               "val_loss", "val_accuracy"] + CV_FIELDS +
              ["is_best", "seconds", "eval_seconds", "timestamp"])
    f = open(path, "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); f.flush()

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.time()
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            train_s = time.time() - self.t0
            state["global_epoch"] += 1
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            t1 = time.time()
            extra = epoch_hook(self.model, state["global_epoch"]) if epoch_hook else {}
            w.writerow({
                "global_epoch": state["global_epoch"], "stage": state["stage"],
                "stage_epoch": epoch + 1, "lr": lr,
                "loss": logs.get("loss"), "accuracy": logs.get("accuracy"),
                "val_loss": logs.get("val_loss"), "val_accuracy": logs.get("val_accuracy"),
                **extra,
                "seconds": round(train_s, 1), "eval_seconds": round(time.time() - t1, 1),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
            f.flush()
    return EpochLogger()


def load_clean_images(np, df, root, image_size):
    """Cached frames as uint8 (N,S,S,3), read exactly as SaFariVWW.diagnose does."""
    from PIL import Image
    root = Path(root)
    arr = np.empty((len(df), image_size, image_size, 3), dtype=np.uint8)
    for i, p in enumerate(df["path"]):
        with Image.open(root / p) as im:
            im = im.convert("RGB")
            if im.size != (image_size, image_size):
                raise SystemExit(f"{p}: expected {image_size}x{image_size} cache image, got {im.size}")
            arr[i] = np.asarray(im)
    return arr


def make_predictor(tf, np, model, batch=64):
    """Inference function for `model`: float32 (N,S,S,3) in [0,1] -> (N,2) softmax.

    One compiled tf.function per model, fed small batches. Calling
    model.predict() repeatedly in one process leaks GPU memory (a new input
    pipeline + a full-chunk GPU copy per call), which ran a 4 GB GPU out of
    memory after ~8 epochs. Used for the per-epoch scoring AND the final
    diagnose, so both compute exactly the same numbers."""
    fn = tf.function(lambda x: model(x, training=False), reduce_retracing=True)

    def predict(images):
        out = [fn(tf.constant(images[s:s + batch])).numpy()
               for s in range(0, len(images), batch)]
        return np.concatenate(out) if out else np.empty((0, 2), np.float32)
    return predict


def predict_animal(np, predictor, images, chunk=512):
    """P(animal) for uint8 images; same /255 and chunking as SaFariVWW.diagnose."""
    out = np.empty(len(images), dtype=np.float32)
    for s in range(0, len(images), chunk):
        x = images[s:s + chunk].astype(np.float32) / 255.0
        out[s:s + chunk] = predictor(x)[:, 1]
    return out


def clean_val_metrics(np, binary_metrics, y, p):
    """SaFariVWW metrics at threshold 0.5 + binary cross-entropy (Keras clipping)."""
    m = binary_metrics(y, p, 0.5)
    q = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    m["loss"] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return m


class BestTracker:
    """Keeps the best value of one metric (strictly better; first epoch wins ties)."""
    def __init__(self, metric):
        self.metric, self.lower = metric, metric == "loss"
        self.value, self.epoch = None, None
    def update(self, value, epoch):
        better = (self.value is None or
                  (value < self.value if self.lower else value > self.value))
        if better:
            self.value, self.epoch = value, epoch
        return better


def attach_callbacks(model, callbacks, fit_verbose=None):
    """Make every model.fit(...) call (made by MLPerf's train_epochs) also use
    our callbacks. Only this model instance is affected; MLPerf code unchanged."""
    original_fit = model.fit

    def fit_with_logging(*args, **kwargs):
        kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + list(callbacks)
        if fit_verbose is not None:
            kwargs["verbose"] = fit_verbose
        return original_fit(*args, **kwargs)
    model.fit = fit_with_logging
    return model


# ---------------------------------------------------------------- model

MLPERF_INPUT_LINE = "input_shape = [96,96,3]"


def mlperf_mobilenet_v1(mlperf_dir, image_size):
    """MLPerf's mobilenet_v1 at any input size.

    96: their vww_model.mobilenet_v1() called unchanged.
    other: their vww_model.py source executed with only its input line
    changed to [N,N,3] (file on disk untouched). Everything else, including
    the final AveragePooling2D (sized from the feature map), is theirs."""
    if image_size == 96:
        from vww_model import mobilenet_v1
        return mobilenet_v1()
    path = Path(mlperf_dir) / "vww_model.py"
    src = path.read_text()
    if src.count(MLPERF_INPUT_LINE) != 1:
        raise SystemExit(f"cannot find '{MLPERF_INPUT_LINE}' once in {path}")
    src = src.replace(MLPERF_INPUT_LINE, f"input_shape = [{image_size},{image_size},3]")
    ns = {"__name__": f"vww_model_{image_size}", "__file__": str(path)}
    exec(compile(src, str(path), "exec"), ns)
    return ns["mobilenet_v1"]()


def load_init_weights(tf, model, init_path):
    """Copy all weights of a saved MLPerf model (any resolution) into `model`.
    Layer-by-layer shapes must match exactly (they do across resolutions)."""
    src = tf.keras.models.load_model(init_path)
    sw, dw = src.get_weights(), model.get_weights()
    if len(sw) != len(dw) or any(a.shape != b.shape for a, b in zip(sw, dw)):
        raise SystemExit(f"{init_path}: weights do not match the mobilenet_v1 architecture")
    model.set_weights(sw)
    return src.input_shape[1]


# ---------------------------------------------------------------- data

def load_split(pd, csv_path):
    df = pd.read_csv(csv_path, dtype={"path": str, "label": str})
    if list(df.columns) != ["path", "label"]:
        raise ValueError(f"unexpected columns in {csv_path}: {list(df.columns)}")
    if not set(df["label"]) <= set(CLASSES):
        raise ValueError(f"unexpected labels in {csv_path}: {set(df['label'])}")
    if df["path"].duplicated().any():
        raise ValueError(f"duplicate paths in {csv_path}")
    return df


def mlperf_datagen(tf):
    """Exactly train_vww.py's ImageDataGenerator, minus validation_split
    (we supply our own validation set)."""
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=.1,
        horizontal_flip=True,
        rescale=1. / 255)


def flow(datagen, df, directory, image_size, batch_size, seed, shuffle=True):
    return datagen.flow_from_dataframe(
        df, directory=str(directory), x_col="path", y_col="label",
        classes=CLASSES, class_mode="categorical",
        target_size=(image_size, image_size), batch_size=batch_size,
        color_mode="rgb", shuffle=shuffle, seed=seed)


# ---------------------------------------------------------------- main

def main(argv=None):
    a = parse_args(argv)
    run_dir = Path(a.logs_dir) / a.run_name
    run_ckpt = Path(a.ckpt_dir) / a.run_name
    if (run_dir / "summary.json").exists():
        raise SystemExit(f"run '{a.run_name}' already finished in {run_dir}; pick another --run-name")
    if (run_dir / "epochs.csv").exists() or run_ckpt.exists():
        raise SystemExit(f"run '{a.run_name}' has an unfinished earlier attempt; delete it first:\n"
                         f"  rm -r '{run_dir}' '{run_ckpt}'")
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(sys.stdout, run_dir / "console.log")
    sys.stderr = Tee(sys.stderr, run_dir / "console.log")

    sys.path.insert(0, str(Path(a.mlperf_dir)))
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import train_vww                                      # MLPerf, unmodified
    from safari_tools import SaFariAnnotation
    from safari_vww import SaFariVWW

    if Path(train_vww.__file__).resolve().parent != Path(a.mlperf_dir).resolve():
        raise SystemExit(f"imported the wrong train_vww: {train_vww.__file__}")

    # ---- reproducibility: same config -> same result ----
    tf.keras.utils.set_random_seed(a.seed)        # python random, numpy, tensorflow
    if a.deterministic:
        tf.config.experimental.enable_op_determinism()   # deterministic GPU/CPU kernels
    print(f"seed {a.seed}  deterministic {a.deterministic}")

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)

    # ---- data (exact lists used are saved with the run) ----
    sets = {"train": a.train_csv, "val": a.val_csv}
    if a.test_csv:
        sets["test"] = a.test_csv
    limits = {"train": a.limit_train, "val": a.limit_val, "test": a.limit_test}
    dfs, used_csv = {}, {}
    for k, path in sets.items():
        df = load_split(pd, path)
        if limits[k] and limits[k] < len(df):
            df = df.sample(n=limits[k], random_state=a.seed)
        dfs[k] = df
        used_csv[k] = run_dir / "data" / f"{k}.csv"
        df.to_csv(used_csv[k], index=False)
        print(f"{k:5}: {len(df):>7} frames  animal {100*(df.label=='1').mean():.1f}%  <- {path}")

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    config.update({
        "image_size": a.image_size, "batch_size": train_vww.BATCH_SIZE,
        "mlperf_train_vww": train_vww.__file__,
        "tensorflow": tf.__version__, "gpus": [g.name for g in gpus],
        "frames": {k: len(v) for k, v in dfs.items()},
        "started": dt.datetime.now().isoformat(timespec="seconds"),
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    datagen = mlperf_datagen(tf)
    gen_args = dict(image_size=a.image_size, batch_size=train_vww.BATCH_SIZE, seed=a.seed)
    train_generator = flow(datagen, dfs["train"], a.train_images, **gen_args)
    # MLPerf draws validation from the same (augmented) generator; kept identical.
    val_generator = flow(datagen, dfs["val"], a.train_images, **gen_args)
    if train_generator.class_indices != {"0": 0, "1": 1}:
        raise SystemExit(f"unexpected class_indices {train_generator.class_indices}")
    print("class_indices:", train_generator.class_indices)

    # ---- model: fresh (MLPerf) or from a saved model (as train_vww.main argv[1]) ----
    if a.init and a.image_size == 96:
        model = tf.keras.models.load_model(a.init)       # as train_vww.main argv[1]
        print("initialized from:", a.init)
    else:
        model = mlperf_mobilenet_v1(a.mlperf_dir, a.image_size)
        if a.init:
            src_size = load_init_weights(tf, model, a.init)
            print(f"initialized from: {a.init} (trained at {src_size}, weights copied to {a.image_size})")
        else:
            print(f"initialized: fresh MLPerf mobilenet_v1 at {a.image_size}x{a.image_size}")
    shp_in, shp_out = tuple(model.input_shape[1:]), model.output_shape[-1]
    if shp_in != (a.image_size, a.image_size, 3) or shp_out != len(CLASSES):
        raise SystemExit(f"model shape {shp_in} -> {shp_out} does not match the VWW contract")
    model.summary()

    # ---- per-epoch clean-val scoring + checkpoints (read-only w.r.t. training) ----
    t0 = time.time()
    val_images = load_clean_images(np, dfs["val"], a.train_images, a.image_size)
    val_labels = dfs["val"]["label"].astype(int).to_numpy()
    preds_dir = run_dir / "val_preds"; preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / "labels.npy", val_labels)
    print(f"clean val loaded: {len(val_images)} frames in {time.time() - t0:.0f}s")

    epochs_dir = run_ckpt / "epochs"
    (epochs_dir if a.epoch_checkpoints else run_ckpt).mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_ckpt / "best.h5", run_ckpt / "last.h5"
    tracker = BestTracker(a.select)
    predictors = {}                                  # one compiled predictor per model object

    def epoch_hook(m, ep):
        if id(m) not in predictors:
            predictors[id(m)] = make_predictor(tf, np, m)
        p = predict_animal(np, predictors[id(m)], val_images)
        np.save(preds_dir / f"ep{ep:03d}.npy", p)
        met = clean_val_metrics(np, SaFariVWW._binary_metrics, val_labels, p)
        if a.epoch_checkpoints:
            m.save(str(epochs_dir / f"ep{ep:03d}.h5"), include_optimizer=False)
        is_best = tracker.update(met[a.select], ep)
        if is_best:
            m.save(str(best_path), include_optimizer=False)
        print(f"  [clean val] ep {ep:3d}  bal {met['balanced_accuracy']:.4f}  "
              f"auc {met['auc']:.4f}  loss {met['loss']:.4f}"
              + ("  <- best" if is_best else ""))
        return {**{f"cv_{k}": met[k] for k in
                   ("loss", "accuracy", "balanced_accuracy", "auc", "recall_animal",
                    "recall_empty", "precision_animal", "f1_animal")},
                "is_best": int(is_best)}

    state = {"global_epoch": 0, "stage": 0}
    attach_callbacks(model, [make_epoch_logger(tf, run_dir / "epochs.csv", state, epoch_hook)],
                     a.fit_verbose)

    # ---- train: MLPerf's own function, one call per stage ----
    t_start = time.time()
    for k, (epochs, lr) in enumerate(a.stages, 1):
        state["stage"] = k
        print(f"=== stage {k}/{len(a.stages)}: {epochs} epochs @ lr {lr} ===")
        model = train_vww.train_epochs(model, train_generator, val_generator, epochs, lr)
        if a.save_stages:
            p = run_ckpt / f"stage{k}.h5"
            model.save(str(p), include_optimizer=False); print("saved stage model:", p)
    train_minutes = round((time.time() - t_start) / 60, 1)

    model.save(str(last_path), include_optimizer=False)
    print(f"saved: {last_path}  (last, epoch {state['global_epoch']})")
    print(f"saved: {best_path}  (best clean-val {a.select} {tracker.value:.4f}, epoch {tracker.epoch})")
    del val_images, predictors

    # ---- evaluation of best and last (does not affect training) ----
    # val: threshold 0.5 and the val-optimal threshold; test: 0.5 and that SAME val threshold.
    models = {"best": tf.keras.models.load_model(str(best_path), compile=False),
              "last": tf.keras.models.load_model(str(last_path), compile=False)}
    final_predictors = {n: make_predictor(tf, np, m) for n, m in models.items()}
    KEYS = ("accuracy", "balanced_accuracy", "auc", "recall_animal", "recall_empty",
            "precision_animal", "f1_animal")
    results, metrics, thresholds = {}, {n: {} for n in models}, {}

    def diag(vww, name, model_, csv_path, root, split, threshold=0.5, tval=False):
        save = not tval
        tag = f"{name}_{split}" + ("_tval" if tval else "")
        r = vww.diagnose(final_predictors[name], csv_path, root, name=f"{a.run_name}-{tag}",
                         threshold=threshold, image_size=a.image_size,
                         out_csv=run_dir / f"diag_{tag}_frames.csv" if save else None,
                         verbose=save)
        results[tag] = r
        metrics[name][split + ("_at_val_threshold" if tval else "")] = \
            {k: r["overall"][k] for k in KEYS}
        return r

    if a.ann_train:
        vww = SaFariVWW(SaFariAnnotation.from_json(a.ann_train))
        for n, m in models.items():
            r = diag(vww, n, m, used_csv["val"], a.train_images, "val")
            thresholds[n] = r["best_threshold"]["threshold"]
            metrics[n]["val_at_val_threshold"] = {k: r["best_threshold"][k] for k in KEYS}
        del vww
    if a.ann_test and "test" in dfs:
        vww = SaFariVWW(SaFariAnnotation.from_json(a.ann_test))
        for n, m in models.items():
            diag(vww, n, m, used_csv["test"], a.test_images, "test")
            if n in thresholds:
                diag(vww, n, m, used_csv["test"], a.test_images, "test",
                     threshold=thresholds[n], tval=True)
    for tag, r in results.items():
        (run_dir / f"diag_{tag}.json").write_text(json.dumps(r, indent=2, default=float))

    summary = {
        "run_name": a.run_name, "image_size": a.image_size, "init": a.init, "seed": a.seed,
        "schedule": a.stages, "train_minutes": train_minutes, "epochs": state["global_epoch"],
        "selection": {"metric": a.select, "on": "clean val (no augmentation)",
                      "best_epoch": tracker.epoch, "best_value": tracker.value},
        "models": {"best": str(best_path), "last": str(last_path),
                   "epochs_dir": str(epochs_dir) if a.epoch_checkpoints else None},
        "val_threshold": thresholds,
        "metrics": metrics,
        "run_dir": str(run_dir),
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("summary:", run_dir / "summary.json")
    print(f"best = epoch {tracker.epoch} (clean-val {a.select}), last = epoch {state['global_epoch']}")
    for n in metrics:
        for split, m in metrics[n].items():
            print(f"  {n:4} {split:24}: balanced {m['balanced_accuracy']:.4f}  "
                  f"AUC {m['auc']:.4f}  rec animal {m['recall_animal']:.4f}  "
                  f"rec empty {m['recall_empty']:.4f}")


if __name__ == "__main__":
    main()