"""Train a transfer-learning classifier using an eligible split manifest."""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from backend.app.preprocess import CLASSES, INPUT_SIZE, MODELS, PREPROCESSING
from backend.app.registry import file_hash
from training.preprocess import PILOT_SCOPE, dataset, partition_rows


def build_model(model_name, dropout, weights="imagenet", head_units=0, focal_crop=False):
    import tensorflow as tf

    constructors = dict(zip(MODELS, (tf.keras.applications.MobileNetV2,
                                    tf.keras.applications.ResNet50,
                                    tf.keras.applications.EfficientNetB0)))
    backbone = constructors[model_name](weights=weights, include_top=False, input_shape=(*INPUT_SIZE, 3))
    backbone.trainable = False
    inputs = tf.keras.Input((*INPUT_SIZE, 3))
    pixels = inputs
    if focal_crop:
        height, width = INPUT_SIZE
        pixels = tf.keras.layers.Cropping2D(((height // 2, 0), (width // 4, width // 4)))(pixels)
        pixels = tf.keras.layers.Resizing(height, width)(pixels)
    features = backbone(pixels, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="road_features")(features)
    hidden = tf.keras.layers.Dense(head_units, activation="relu")(pooled) if head_units else pooled
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(tf.keras.layers.Dropout(dropout)(hidden))
    return tf.keras.Model(inputs, outputs), backbone


def cache_head_features(model, train, validation, passes, batch_size, seed):
    import tensorflow as tf

    pooled_layer = model.get_layer("road_features")
    extractor = tf.keras.Model(model.inputs, pooled_layer.output)
    head = tf.keras.Sequential(model.layers[model.layers.index(pooled_layer) + 1:])

    @tf.function(reduce_retracing=True)
    def encode(pixels):
        return extractor(pixels, training=False)

    def features(data, repeats):
        vectors, targets = [], []
        for _ in range(repeats):
            for pixels, labels in data:
                vectors.append(encode(pixels).numpy())
                targets.append(labels.numpy())
        return tf.data.Dataset.from_tensor_slices((tf.concat(vectors, 0), tf.concat(targets, 0)))

    print(f"Extracting frozen training features across {passes} passes", flush=True)
    training = features(train, passes)
    training = training.shuffle(int(training.cardinality()), seed=seed).batch(batch_size)
    print("Extracting frozen validation features", flush=True)
    validation_features = features(validation, 1).batch(batch_size)
    return head, training, validation_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/default.json")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/models"))
    parser.add_argument("--capture-date-pilot", action="store_true")
    args = parser.parse_args()
    try:
        partitions = partition_rows(args.manifest, pilot=args.capture_date_pilot)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    target = args.output_dir / f"{args.model}.keras"
    if target.exists() or target.with_suffix(".json").exists():
        parser.error("Model artifact already exists. Use a new output directory.")
    import tensorflow as tf

    config = json.loads(args.config.read_text())
    monitor = config.get("selection_metric", "val_loss")
    if monitor not in {"val_accuracy", "val_loss"}:
        parser.error("selection_metric must be val_accuracy or val_loss.")
    mode = "max" if monitor == "val_accuracy" else "min"
    better = max if mode == "max" else min
    seed = config["seed"]
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    cache_decoded = args.capture_date_pilot or config.get("cache_decoded", False)
    train = dataset(partitions["train"], args.model, config["batch_size"], training=True, seed=seed,
                    cache_decoded=cache_decoded)
    validation = None if args.capture_date_pilot else dataset(
        partitions["validation"], args.model, config["batch_size"], cache_decoded=cache_decoded)
    model, backbone = build_model(args.model, config["dropout"], head_units=config.get("head_units", 0),
                                  focal_crop=config.get("focal_crop", False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        directory = Path(temporary)

        def fit(phase, learning_rate, epochs, network, inputs, validation_data):
            network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            checkpoint = directory / f"{phase}.weights.h5"
            history = network.fit(inputs, validation_data=validation_data, epochs=epochs, shuffle=False, callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=config["patience"]),
                tf.keras.callbacks.ReduceLROnPlateau(patience=max(1, config["patience"] // 2)),
            ]).history
            epoch = better(range(len(history[monitor])), key=history[monitor].__getitem__)
            network.load_weights(checkpoint)
            selected = dict(phase=phase, epoch=epoch + 1, metric=monitor,
                            val_accuracy=history["val_accuracy"][epoch], val_loss=history["val_loss"][epoch])
            return selected, checkpoint, history

        if args.capture_date_pilot:
            # A fixed schedule keeps the only held-out capture date out of model selection.
            counts = Counter(row["verified_quality_label"] for row in partitions["train"])
            weights_by_class = {index: len(partitions["train"]) / (len(CLASSES) * counts[label])
                                for index, label in enumerate(CLASSES)}
            model.compile(optimizer=tf.keras.optimizers.Adam(config["head_lr"]),
                          loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            history = model.fit(train, epochs=config["head_epochs"], shuffle=False,
                                class_weight=weights_by_class).history
        else:
            head, head_train, head_validation = model, train, validation
            if config.get("feature_passes", 0):
                head, head_train, head_validation = cache_head_features(
                    model, train, validation, config["feature_passes"], config["batch_size"], seed)
            selected, weights, history = fit(
                "head", config["head_lr"], config["head_epochs"], head, head_train, head_validation)
            if head is not model:
                weights = directory / "head-full-model.weights.h5"
                model.save_weights(weights)
        histories = {"head": history}
        if not args.capture_date_pilot and config["fine_tune_epochs"] > 0 and config["fine_tune_layers"] > 0:
            backbone.trainable = True
            cutoff = max(0, len(backbone.layers) - config["fine_tune_layers"])
            for index, layer in enumerate(backbone.layers):
                layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
            tuned_selection, tuned, histories["fine_tune"] = fit(
                "fine_tune", config["fine_tune_lr"], config["fine_tune_epochs"], model, train, validation)
            if better(selected[monitor], tuned_selection[monitor]) != selected[monitor]:
                selected, weights = tuned_selection, tuned
            model.load_weights(weights)
        staged = directory / target.name
        model.save(staged)
        digest = file_hash(staged)
        metadata = dict(model=args.model, version=digest[:12], sha256=digest, classes=list(CLASSES),
                        input_size=list(INPUT_SIZE), preprocessing=PREPROCESSING, config=config,
                        training_manifest_sha256=file_hash(args.manifest), history=histories,
                        training_examples=len(partitions["train"]),
                        partition_sources={split: dict(Counter(r.get("source") or "unspecified" for r in rows))
                                           for split, rows in partitions.items()})
        if args.capture_date_pilot:
            metadata.update(use_scope=PILOT_SCOPE)
        else:
            metadata["validation_selection"] = selected
        with target.with_suffix(".json").open("x") as file:
            json.dump(metadata, file, indent=2)
        staged.rename(target)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
