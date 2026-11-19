"""Save class probabilities and pooled features of saved models for chosen partitions.

Used for validation-only error analysis, ensembling and threshold calibration.
The test partition is refused unless --allow-test is given.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from backend.app.preprocess import MODELS, uses_focal_crop
from training.preprocess import dataset, partition_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if "test" in args.splits and not args.allow_test:
        parser.error("The test partition is reserved for the final evaluation; pass --allow-test explicitly.")
    import tensorflow as tf

    # ---- For each model, save its feature vectors and probabilities for the chosen splits ----
    partitions = partition_rows(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        metadata = json.loads((args.model_dir / f"{name}.json").read_text())
        focal = uses_focal_crop(metadata["preprocessing"])
        model = tf.keras.models.load_model(args.model_dir / f"{name}.keras", compile=False)
        # A model with two outputs: the pooled "road_features" vector and the final probabilities.
        both = tf.keras.Model(model.inputs, [model.get_layer("road_features").output, model.output])
        for split in args.splits:
            rows = partitions[split]
            features, probabilities = [], []
            for pixels, _ in dataset(rows, name, 32, focal=focal):
                pooled, scores = both(pixels, training=False)
                features.append(np.asarray(pooled))
                probabilities.append(np.asarray(scores))
            # .npz = compressed NumPy archive; image_ids keep rows in a known order, sha256 ties it to the model.
            np.savez_compressed(args.out / f"{name}-{split}.npz", features=np.concatenate(features),
                                probabilities=np.concatenate(probabilities),
                                image_ids=np.array([r["image_id"] for r in rows]), model_sha256=metadata["sha256"])
            print(f"Saved {name} {split}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
