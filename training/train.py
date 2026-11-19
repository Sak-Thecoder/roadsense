"""Train a transfer-learning classifier using an eligible split manifest.

Overview of the training strategy (transfer learning in two phases):
  1. Head phase: the ImageNet-pre-trained backbone (MobileNetV2 / ResNet50 /
     EfficientNetB0) is frozen, and only the new small classifier "head" on top
     is trained. This is fast and safe because the head starts from random weights.
  2. Fine-tune phase: the backbone layers are unfrozen and the whole network is
     trained with a small learning rate, so the features adapt to road surfaces.

After each epoch the model is scored on the validation set, and the best epoch
(by the `selection_metric`, e.g. macro F1) is kept. The test set is never used here.

Example:
    python -m training.train --manifest work/approved_splits.csv --model resnet50 \
        --config training/configs/quality_v3.json --output-dir work/run/models
"""
import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

from backend.app.preprocess import CLASSES, FOCAL_PREPROCESSING, INPUT_SIZE, MODELS, PREPROCESSING
from backend.app.registry import file_hash
from training.preprocess import PILOT_SCOPE, dataset, partition_rows


# Metrics that can be used to pick the best epoch, and whether higher ("max") or lower ("min") is better.
MONITORS = {"val_accuracy": "max", "val_macro_f1": "max", "val_loss": "min"}


# "grade": train directly on Good/Moderate/Poor.
# "surface_x_level": train on finer classes such as "asphalt:good" or "unpaved:bad".
LABEL_SCHEMES = ("grade", "surface_x_level")


def fine_label(row, scheme):
    """Training target. surface_x_level keeps StreetSurfaceVis's per-surface grading scale apart.

    StreetSurfaceVis grades each surface type on its own scale ("good" asphalt
    looks very different from "good" gravel). Giving the model the surface type
    as part of the label lets it learn these scales separately.
    Example results: "Good" (grade scheme) or "asphalt:excellent" (surface_x_level).
    """
    grade = row["verified_quality_label"]
    if scheme == "grade":
        return grade
    level = row.get("original_label") if row.get("source") == "streetSurfaceVis" else grade.lower()
    return f"{row.get('surface_type') or 'local'}:{level or grade.lower()}"


def label_space(rows, scheme):
    """Training classes and the grade each one belongs to, in a fixed order.

    Returns (classes, grade_of). Example for the fine scheme:
        classes  = ["asphalt:bad", "asphalt:excellent", ...]
        grade_of = [2,             0,                   ...]   # 0=Good, 1=Moderate, 2=Poor
    """
    if scheme == "grade":
        return list(CLASSES), list(range(len(CLASSES)))
    grades = {}
    for row in rows:
        grades.setdefault(fine_label(row, scheme), CLASSES.index(row["verified_quality_label"]))
    classes = sorted(grades)
    return classes, [grades[c] for c in classes]


def build_model(model_name, dropout, weights="imagenet", head_units=0, outputs=len(CLASSES)):
    """Create  input -> pre-trained backbone -> average pooling -> [Dense] -> Dropout -> softmax.

    Returns (model, backbone); the backbone is returned separately so it can be unfrozen later.
    """
    import tensorflow as tf

    constructors = dict(zip(MODELS, (tf.keras.applications.MobileNetV2,
                                    tf.keras.applications.ResNet50,
                                    tf.keras.applications.EfficientNetB0)))
    # include_top=False removes the original 1000-class ImageNet classifier; we add our own.
    backbone = constructors[model_name](weights=weights, include_top=False, input_shape=(*INPUT_SIZE, 3))
    backbone.trainable = False  # frozen for the head phase
    inputs = tf.keras.Input((*INPUT_SIZE, 3))
    # Any focal crop is applied to the full-resolution image during decoding, not here.
    # training=False keeps BatchNormalization layers in inference mode even during fine-tuning,
    # which is the recommended practice for transfer learning.
    features = backbone(inputs, training=False)
    # Average each feature map to one number -> a single feature vector per image.
    # The layer is named so other scripts can extract these "road_features".
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="road_features")(features)
    hidden = tf.keras.layers.Dense(head_units, activation="relu")(pooled) if head_units else pooled
    # Dropout randomly turns off neurons during training to reduce overfitting;
    # softmax turns the outputs into probabilities that sum to 1.
    scores = tf.keras.layers.Dense(outputs, activation="softmax")(tf.keras.layers.Dropout(dropout)(hidden))
    return tf.keras.Model(inputs, scores), backbone


def grade_output(model, grade_of):
    """Append a fixed layer that sums fine-class probabilities into Good/Moderate/Poor.

    It is a Dense layer with a 0/1 weight matrix and no bias. For example, if
    "asphalt:excellent" and "asphalt:good" both belong to Good, then
    P(Good) = P(asphalt:excellent) + P(asphalt:good) + ... The saved model therefore
    still outputs exactly three grades, which is what the API expects.
    """
    import numpy as np
    import tensorflow as tf

    # Shape (number_of_fine_classes, 3). Row i has a single 1 in the column of its grade.
    matrix = np.zeros((len(grade_of), len(CLASSES)), np.float32)
    matrix[np.arange(len(grade_of)), grade_of] = 1
    layer = tf.keras.layers.Dense(len(CLASSES), use_bias=False, trainable=False, name="grade_probabilities")
    wrapped = tf.keras.Model(model.inputs, layer(model.output))
    layer.set_weights([matrix])
    return wrapped


def balanced_class_weights(rows, grade_of=None):
    """Inverse grade frequency; with fine classes, each class takes the weight of its grade.

    weight(grade) = total_images / (3 * images_of_that_grade).
    Good is the most common grade, so it gets a weight below 1. Moderate and Poor are
    rarer and get larger weights, so mistakes on them cost more during training.
    """
    counts = Counter(row["verified_quality_label"] for row in rows)
    weights = [len(rows) / (len(CLASSES) * counts[label]) for label in CLASSES]
    return {index: weights[grade] for index, grade in enumerate(grade_of or range(len(CLASSES)))}


def macro_f1_callback(validation, grade_of=None):
    """Add grade-level val_macro_f1 and val_grade_accuracy to the epoch logs.

    List it before callbacks that monitor these values.

    Keras has no built-in macro F1, so after every epoch this callback predicts the
    whole validation set, converts fine classes back to the three grades, and
    computes macro F1 with scikit-learn.
    """
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import f1_score

    grades = np.asarray(grade_of if grade_of is not None else range(len(CLASSES)))

    class MacroF1(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            actual, predicted = [], []
            for inputs, labels in validation:
                scores = np.asarray(self.model(inputs, training=False))
                # Sum fine-class probabilities into their grades. np.add.at adds column j of
                # `scores` into grade column grades[j], correctly handling repeated grades.
                summed = np.zeros((len(scores), len(CLASSES)))
                np.add.at(summed.T, grades, scores.T)
                predicted.append(summed.argmax(axis=1))
                labels = np.asarray(labels)
                # Labels are either integers or one-hot vectors (with label smoothing); convert to grades.
                actual.append(grades[labels.argmax(axis=1) if labels.ndim == 2 else labels])
            actual, predicted = np.concatenate(actual), np.concatenate(predicted)
            score = f1_score(actual, predicted, labels=range(len(CLASSES)), average="macro", zero_division=0)
            accuracy = float((actual == predicted).mean())
            if logs is not None:
                # Writing into `logs` makes these values visible to ModelCheckpoint/EarlyStopping.
                logs["val_macro_f1"] = float(score)
                logs["val_grade_accuracy"] = accuracy
            print(f" - val_macro_f1: {score:.4f} - val_grade_accuracy: {accuracy:.4f}", flush=True)

    return MacroF1()


def cache_head_features(model, train, validation, passes, batch_size, seed):
    """Speed up the head phase by running the frozen backbone only once per image.

    While the backbone is frozen its output never changes, so we compute the pooled
    feature vectors once (or `passes` times with different random augmentations)
    and train just the small head on those vectors. This is many times faster
    than pushing every image through the full CNN every epoch.
    """
    import tensorflow as tf

    pooled_layer = model.get_layer("road_features")
    extractor = tf.keras.Model(model.inputs, pooled_layer.output)  # image -> feature vector
    # The head = every layer after "road_features" (Dense/Dropout/softmax).
    head = tf.keras.Sequential(model.layers[model.layers.index(pooled_layer) + 1:])

    # tf.function compiles the call into a fast TensorFlow graph.
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
    # ---- 1. Read command-line arguments and the dataset manifest ----
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
    # Never overwrite an existing trained model by accident.
    if target.exists() or target.with_suffix(".json").exists():
        parser.error("Model artifact already exists. Use a new output directory.")
    import tensorflow as tf

    # ---- 2. Read and validate the JSON config (see training/configs/*.json) ----
    config = json.loads(args.config.read_text())
    monitor = config.get("selection_metric", "val_loss")
    if monitor not in MONITORS:
        parser.error(f"selection_metric must be one of {', '.join(MONITORS)}.")
    mode = MONITORS[monitor]
    focal = bool(config.get("focal_crop", False))
    augmentation = config.get("augmentation", "basic")
    class_weight = config.get("class_weight")
    if class_weight not in {None, "balanced"}:
        parser.error("class_weight must be null or balanced.")
    scheme = config.get("label_scheme", "grade")
    if scheme not in LABEL_SCHEMES or (args.capture_date_pilot and scheme != "grade"):
        parser.error(f"label_scheme must be one of {', '.join(LABEL_SCHEMES)} (grade in pilot mode).")
    smoothing = float(config.get("label_smoothing", 0))

    # ---- 3. Decide the training classes (3 grades or fine surface:level classes) ----
    fine_classes, grade_of = label_space(partitions["train"], scheme)
    # For each grade, its first fine class, used for validation images whose fine class never appears in train.
    fallback = {grade: fine_classes[grade_of.index(grade)] for grade in set(grade_of)}

    def label_index(row):
        # A fine class absent from training is scored as its grade's first fine class (grade metrics unaffected).
        label = fine_label(row, scheme)
        return fine_classes.index(label if label in fine_classes else fallback[CLASSES.index(
            row["verified_quality_label"])])
    # `better` picks the best value: max() for accuracy/F1, min() for loss.
    better = max if mode == "max" else min

    # ---- 4. Make training reproducible and build the input pipelines ----
    seed = config["seed"]
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    cache_decoded = args.capture_date_pilot or config.get("cache_decoded", False)
    train = dataset(partitions["train"], args.model, config["batch_size"], training=True, seed=seed,
                    cache_decoded=cache_decoded, focal=focal, augmentation=augmentation, label_index=label_index)
    validation = None if args.capture_date_pilot else dataset(
        partitions["validation"], args.model, config["batch_size"], cache_decoded=cache_decoded, focal=focal,
        label_index=label_index)
    loss = "sparse_categorical_crossentropy"
    if smoothing:
        # Label smoothing replaces a hard target like [0, 1, 0] with a softer one like
        # [0.05, 0.9, 0.05], so the model isn't punished as hard for borderline/noisy labels.
        # It requires one-hot labels, so integer labels are converted with tf.one_hot.
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)
        train, validation = (None if data is None else data.map(
            lambda pixels, labels: (pixels, tf.one_hot(labels, len(fine_classes)))) for data in (train, validation))
    weights_by_class = balanced_class_weights(partitions["train"], grade_of) if class_weight else None
    if weights_by_class:
        print(f"Class weights: { {fine_classes[i]: round(w, 3) for i, w in weights_by_class.items()} }", flush=True)
    model, backbone = build_model(args.model, config["dropout"], head_units=config.get("head_units", 0),
                                  outputs=len(fine_classes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Checkpoints go into a temporary folder that is deleted automatically at the end.
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        directory = Path(temporary)

        def fit(phase, learning_rate, epochs, network, inputs, validation_data):
            """Train `network`, keep the best epoch's weights, and return a summary of that epoch."""
            network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss, metrics=["accuracy"])
            checkpoint = directory / f"{phase}.weights.h5"
            history = network.fit(inputs, validation_data=validation_data, epochs=epochs, shuffle=False,
                                  class_weight=weights_by_class, callbacks=[
                # Must come first so val_macro_f1 exists when the callbacks below read it.
                macro_f1_callback(validation_data, grade_of),
                # Save weights only when the monitored metric improves.
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=True),
                # Stop if there is no improvement for `patience` epochs.
                tf.keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=config["patience"]),
                # Reduce the learning rate when progress stalls for half that long.
                tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode=mode,
                                                     patience=max(1, config["patience"] // 2)),
            ]).history
            # Index of the best epoch according to the monitored metric.
            epoch = better(range(len(history[monitor])), key=history[monitor].__getitem__)
            network.load_weights(checkpoint)
            selected = dict(phase=phase, epoch=epoch + 1, metric=monitor,
                            **{key: history[key][epoch] for key in (*MONITORS, "val_grade_accuracy")})
            return selected, checkpoint, history

        # ---- 5. Phase 1: train the head with the backbone frozen ----
        if args.capture_date_pilot:
            # A fixed schedule keeps the only held-out capture date out of model selection.
            weights_by_class = balanced_class_weights(partitions["train"])
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
                # The head shares its layers with `model`, so saving `model` captures the trained head.
                weights = directory / "head-full-model.weights.h5"
                model.save_weights(weights)
        histories = {"head": history}

        # ---- 6. Phase 2: fine-tune the backbone ----
        if not args.capture_date_pilot and config["fine_tune_epochs"] > 0 and config["fine_tune_layers"] > 0:
            backbone.trainable = True
            # Unfreeze only the last `fine_tune_layers` layers (a large number = the whole network).
            cutoff = max(0, len(backbone.layers) - config["fine_tune_layers"])
            for index, layer in enumerate(backbone.layers):
                # BatchNormalization layers stay frozen: updating their statistics on small
                # batches would damage the pre-trained features.
                layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
            tuned_selection, tuned, histories["fine_tune"] = fit(
                "fine_tune", config["fine_tune_lr"], config["fine_tune_epochs"], model, train, validation)
            # Keep the fine-tuned weights only if they beat the head-only result on validation.
            if better(selected[monitor], tuned_selection[monitor]) != selected[monitor]:
                selected, weights = tuned_selection, tuned
            model.load_weights(weights)

        # ---- 7. Save the model and its metadata (.json) ----
        staged = directory / target.name
        (model if scheme == "grade" else grade_output(model, grade_of)).save(staged)
        digest = file_hash(staged)
        # The API checks this metadata before using the model (classes, input size, preprocessing, sha256).
        metadata = dict(model=args.model, version=digest[:12], sha256=digest, classes=list(CLASSES),
                        input_size=list(INPUT_SIZE), preprocessing=FOCAL_PREPROCESSING if focal else PREPROCESSING,
                        config=config,
                        training_manifest_sha256=file_hash(args.manifest), history=histories,
                        training_examples=len(partitions["train"]), label_scheme=scheme,
                        training_classes=fine_classes,
                        partition_sources={split: dict(Counter(r.get("source") or "unspecified" for r in rows))
                                           for split, rows in partitions.items()})
        if args.capture_date_pilot:
            metadata.update(use_scope=PILOT_SCOPE)
        else:
            metadata["validation_selection"] = selected
        # "x" mode fails if the file already exists, so we never overwrite.
        with target.with_suffix(".json").open("x") as file:
            json.dump(metadata, file, indent=2)
        # Move the finished model into place only after everything else succeeded.
        staged.rename(target)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
