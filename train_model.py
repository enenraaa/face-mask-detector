import argparse
import json
from pathlib import Path

# cspell:ignore argparse keras AUTOTUNE prefetch rglob reshuffle webp softmax filepath onehot crossentropy
import keras
import numpy as np
import tensorflow as tf
from keras import layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face mask detector model.")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root directory.")
    parser.add_argument("--output-dir", default="model", help="Directory for saved model files.")
    parser.add_argument("--img-size", type=int, default=128, help="Square image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def collect_images(dataset_dir: Path) -> tuple[list[str], np.ndarray, list[str]]:
    allowed_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    if len(class_names) < 2:
        raise ValueError("Need at least two class folders inside dataset directory.")

    image_paths: list[str] = []
    labels: list[int] = []

    for class_idx, class_dir in enumerate(class_dirs):
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in allowed_ext:
                image_paths.append(str(file_path))
                labels.append(class_idx)

    if not image_paths:
        raise ValueError(f"No images found in dataset directory: {dataset_dir}")

    return image_paths, np.asarray(labels, dtype=np.int32), class_names


def build_dataset(
    image_paths: list[str],
    labels: np.ndarray,
    img_size: int,
    batch_size: int,
    num_classes: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    def preprocess(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0
        one_hot_label = tf.one_hot(label, depth=num_classes)
        return image, one_hot_label

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size: int, num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    keras.utils.set_random_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    image_paths, labels, class_names = collect_images(dataset_dir)
    num_classes = len(class_names)
    total_samples = len(image_paths)

    indices = np.arange(total_samples)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    val_count = max(1, int(0.2 * total_samples))
    if total_samples - val_count < 1:
        raise ValueError("Dataset is too small after split. Add more images.")

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_paths = [image_paths[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    train_dataset = build_dataset(
        image_paths=train_paths,
        labels=train_labels,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_classes=num_classes,
        training=True,
        seed=args.seed,
    )
    validation_dataset = build_dataset(
        image_paths=val_paths,
        labels=val_labels,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_classes=num_classes,
        training=False,
        seed=args.seed,
    )

    model = build_model(args.img_size, num_classes)
    model.summary()

    model_path = output_dir / "mask_detector.h5"
    labels_path = output_dir / "class_names.json"
    config_path = output_dir / "training_config.json"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    val_loss, val_acc = model.evaluate(validation_dataset, verbose=0)

    with labels_path.open("w", encoding="utf-8") as label_file:
        json.dump(class_names, label_file, indent=2)

    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "class_names": class_names,
            },
            config_file,
            indent=2,
        )

    print(f"Model saved to: {model_path}")
    print(f"Class names saved to: {labels_path}")
    print(f"Training config saved to: {config_path}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
