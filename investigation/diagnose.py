"""Reproduce pilot predictions and inspect diagnostic crops and Grad-CAMs."""
import argparse
import csv
import hashlib
import html
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from backend.app.preprocess import CLASSES, MODELS, INPUT_SIZE, decode, preprocess
from backend.app.registry import Registry, file_hash
from training.preprocess import partition_rows, load_pixels

LAYERS = {
    "mobilenetv2": "out_relu",
    "resnet50": "conv5_block3_out",
    "efficientnetb0": "top_activation",
}


def original(row):
    with zipfile.ZipFile(row["image_path"]) as archive:
        data = archive.read(row["archive_member"])
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"Hash mismatch: {row['image_id']}")
    with Image.open(io.BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def crop_pixels(image):
    width, height = image.size
    crop = image.crop((round(width*.25), round(height*.5), round(width*.75), height))
    return np.asarray(crop.resize(INPUT_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)


def gradcam(model, name, pixels, class_index):
    backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
    probe = tf.keras.Model(backbone.input, [backbone.get_layer(LAYERS[name]).output, backbone.output])
    inputs = tf.convert_to_tensor(preprocess(pixels, name)[None])
    with tf.GradientTape() as tape:
        convolution, features = probe(inputs, training=False)
        tape.watch(convolution)
        output = features
        for layer in model.layers[2:]:
            output = layer(output, training=False)
        score = output[:, class_index]
    gradients = tape.gradient(score, convolution)
    if gradients is None:
        raise RuntimeError(f"No Grad-CAM gradient for {name}")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heat = tf.nn.relu(tf.reduce_sum(weights * convolution, axis=-1))[0].numpy()
    if heat.max() > 0:
        heat /= heat.max()
    return heat, np.asarray(output)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("work/cambodian-pilot/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("work/ml-investigation"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory")
    rows = partition_rows(args.manifest, pilot=True)
    test = rows["test"]
    if len(test) != 21:
        parser.error("Expected exactly the unchanged 21-image pilot test partition")
    registry = Registry(Path("backend/models"), Path("backend/results"))
    models, metadata = {}, {}
    for name in MODELS:
        meta = registry.metadata(name)
        if not meta or meta.get("training_manifest_sha256") != file_hash(args.manifest):
            raise ValueError(f"Artifact or manifest mismatch for {name}")
        models[name] = tf.keras.models.load_model(Path("backend/models") / f"{name}.keras", compile=False)
        metadata[name] = meta

    args.out.mkdir(parents=True)
    (args.out / "images").mkdir()
    predictions = []
    for row in test:
        image = original(row)
        full = load_pixels(row)
        cropped = crop_pixels(image)
        record = dict(image_id=row["original_id"], true_label=row["verified_quality_label"])
        for mode, pixels in (("full", full), ("crop", cropped)):
            for name in MODELS:
                probs = np.asarray(models[name](preprocess(pixels, name)[None], training=False))[0]
                record[f"{mode}_{name}_prediction"] = CLASSES[int(probs.argmax())]
                record[f"{mode}_{name}_confidence"] = float(probs.max())
                for i, label in enumerate(CLASSES):
                    record[f"{mode}_{name}_{label}"] = float(probs[i])
        predictions.append(record)
        thumbnail = image.copy()
        thumbnail.thumbnail((640, 420))
        thumbnail.save(args.out / "images" / f"{row['original_id']}.jpg", quality=85)

    with (args.out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)

    actual = [CLASSES.index(row["true_label"]) for row in predictions]
    stats = {}
    for name in MODELS:
        stats[name] = {}
        for mode in ("full", "crop"):
            predicted = [CLASSES.index(row[f"{mode}_{name}_prediction"]) for row in predictions]
            matrix = confusion_matrix(actual, predicted, labels=range(3))
            stats[name][mode] = {
                "accuracy": accuracy_score(actual, predicted),
                "report": classification_report(actual, predicted, labels=range(3),
                                                 target_names=CLASSES, output_dict=True, zero_division=0),
                "matrix": matrix.tolist(),
                "predicted_counts": dict(Counter(CLASSES[i] for i in predicted)),
            }
            if mode == "full":
                existing = json.loads((Path("backend/results") / f"{name}.json").read_text())
                if matrix.tolist() != existing["confusion_matrix"]:
                    raise ValueError(f"Reproduced matrix differs from saved result for {name}")
                fig, ax = plt.subplots(figsize=(5,4))
                ax.imshow(matrix, cmap="Blues")
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(matrix[i,j]), ha="center", va="center",
                                color="white" if matrix[i,j] > matrix.max()/2 else "black")
                ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES,
                       yticklabels=CLASSES, xlabel="Predicted", ylabel="True", title=name)
                fig.tight_layout()
                fig.savefig(args.out / f"{name}_confusion.png", dpi=150)
                plt.close(fig)

    examples = {}
    for name in MODELS:
        errors = [row for row in predictions if row[f"full_{name}_prediction"] != row["true_label"]]
        # Include one error from each true class when available.
        chosen = []
        for label in CLASSES:
            match = next((r for r in errors if r["true_label"] == label), None)
            if match:
                chosen.append(match)
        examples[name] = [row["image_id"] for row in chosen]
        for record in chosen:
            row = next(r for r in test if r["original_id"] == record["image_id"])
            pixels = load_pixels(row)
            class_index = CLASSES.index(record[f"full_{name}_prediction"])
            heat, check = gradcam(models[name], name, pixels, class_index)
            expected = np.array([record[f"full_{name}_{label}"] for label in CLASSES])
            if not np.allclose(check, expected, atol=1e-5):
                raise ValueError(f"Grad-CAM reconstruction mismatch for {name}")
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(pixels.astype("uint8"))
            ax.imshow(heat, cmap="jet", alpha=.4, extent=(0,224,224,0), interpolation="bilinear")
            ax.set_title(f"{record['image_id']} | true {record['true_label']} | predicted {CLASSES[class_index]}")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(args.out / "images" / f"{name}_{record['image_id']}_gradcam.png", dpi=140)
            plt.close(fig)
    stats["gradcam_examples"] = examples
    (args.out / "metrics.json").write_text(json.dumps(stats, indent=2))

    cards = []
    for record in predictions:
        errors = [n for n in MODELS if record[f"full_{n}_prediction"] != record["true_label"]]
        if not errors:
            continue
        details = "".join(f"<li>{html.escape(n)}: {record[f'full_{n}_prediction']} "
                          f"({record[f'full_{n}_confidence']:.1%})</li>" for n in MODELS)
        overlays = "".join(f'<a href="images/{n}_{record["image_id"]}_gradcam.png">{n} Grad-CAM</a> '
                           for n in MODELS if record["image_id"] in examples[n])
        cards.append(f'<article><img src="images/{record["image_id"]}.jpg" alt="Road {record["image_id"]}">'
                     f'<h2>{record["image_id"]} · true {record["true_label"]}</h2><ul>{details}</ul>{overlays}</article>')
    (args.out / "error_gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RoadSense pilot errors</title>'
        '<style>body{font:16px system-ui;margin:2em;background:#f2f5fa}main{display:grid;'
        'grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1em}article{background:white;'
        'padding:1em;border-radius:10px}img{width:100%;height:260px;object-fit:contain;background:#ddd}'
        'h2{font-size:1em}a{margin-right:1em}</style><h1>Held-out pilot errors</h1>'
        '<p>Original human labels. Grad-CAM overlays show relative sensitivity for the predicted class, '
        'not verified damage localization.</p><main>' + "".join(cards) + '</main>', encoding="utf-8")

    print(json.dumps({name: {mode: {"accuracy": stats[name][mode]["accuracy"],
          "predicted_counts": stats[name][mode]["predicted_counts"]}
          for mode in ("full","crop")} for name in MODELS}, indent=2))


if __name__ == "__main__":
    main()
