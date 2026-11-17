"""Evaluate a saved classifier and record measured metrics and latency."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from backend.app.preprocess import CLASSES, MODELS, preprocess
from backend.app.registry import Registry, file_hash
from backend.app.schemas import Benchmark
from training.preprocess import PILOT_SCOPE, dataset, load_pixels, partition_rows


def evaluate(model, name, rows, metadata, artifact, manifest, split, warmups, repeats):
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    actual, predicted = [], []
    for pixels, labels in dataset(rows, name, 32):
        outputs = np.asarray(model(pixels, training=False))
        actual.extend(labels.numpy().tolist())
        predicted.extend(outputs.argmax(axis=1).tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=range(len(CLASSES)), zero_division=0)
    samples = [load_pixels(rows[i % len(rows)]) for i in range(min(16, len(rows)))]
    timings = []
    for i in range(warmups + repeats):
        before = time.perf_counter()
        np.asarray(model(preprocess(samples[i % len(samples)], name)[None], training=False))
        if i >= warmups:
            timings.append((time.perf_counter() - before) * 1000)
    return Benchmark(evaluation_schema=1, model=name, model_sha256=file_hash(artifact),
        model_version=metadata["version"], classes=list(CLASSES), accuracy=accuracy_score(actual, predicted),
        macro_precision=precision.mean(), macro_recall=recall.mean(), macro_f1=f1.mean(),
        per_class={label: dict(precision=float(precision[i]), recall=float(recall[i]),
                    f1=float(f1[i]), support=int(support[i])) for i, label in enumerate(CLASSES)},
        confusion_matrix=confusion_matrix(actual, predicted, labels=range(len(CLASSES))).tolist(),
        size_mb=artifact.stat().st_size / 1_000_000, parameters=model.count_params(),
        latency_ms=statistics.median(timings), latency_method=(
            f"Median single image preprocessing and synchronous forward pass over {repeats} runs "
            f"after {warmups} warm-ups. Excludes decode, load and network."),
        hardware=f"{platform.platform()} / {platform.processor() or platform.machine()} / TF {tf.__version__}; "
                 f"visible devices: {', '.join(d.device_type for d in tf.config.get_visible_devices())}",
        dataset_size=len(rows), dataset_sha256=file_hash(manifest), split=split,
        use_scope=metadata.get("use_scope"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("backend/models"))
    parser.add_argument("--output-dir", type=Path, default=Path("backend/results"))
    parser.add_argument("--split", choices=("test", "cambodia_holdout"), default="test")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--capture-date-pilot", action="store_true")
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("Timing counts must be positive.")
    try:
        rows = partition_rows(args.manifest, pilot=args.capture_date_pilot)[args.split]
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("No eligible evaluation images in the requested partition.")
    metadata = Registry(args.model_dir, args.output_dir).metadata(args.model)
    if not metadata or metadata.get("training_manifest_sha256") != file_hash(args.manifest):
        parser.error("Model metadata must match the approved training and evaluation manifest.")
    if (metadata.get("use_scope") == PILOT_SCOPE) != args.capture_date_pilot:
        parser.error("Evaluation mode does not match the model artifact.")
    import tensorflow as tf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    artifact = args.model_dir / f"{args.model}.keras"
    model = tf.keras.models.load_model(artifact, compile=False)
    result = evaluate(model, args.model, rows, metadata, artifact, args.manifest, args.split, args.warmups, args.repeats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(np.asarray(result.confusion_matrix), display_labels=CLASSES).plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(args.output_dir / f"{args.model}_confusion.png", dpi=160)
    plt.close()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(("Accuracy", "Macro precision", "Macro recall", "Macro F1"),
           (result.accuracy, result.macro_precision, result.macro_recall, result.macro_f1))
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{args.model}_metrics.png", dpi=160)
    plt.close(fig)
    (args.output_dir / f"{args.model}.json").write_text(result.model_dump_json(indent=2))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
