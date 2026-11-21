import csv
import json
import os

import matplotlib
matplotlib.use("Agg")                    
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers

from data_utils import (CLASS_NAMES, CLASS_NAMES_PATH, IMG_SIZE, MODEL_DIR, MODEL_PATH,
                        PLOTS_DIR, RESULTS_DIR, SEED, TEST_SPLIT_PATH, check_dataset,
                        list_images, make_dataset)

HEAD_EPOCHS = 10          
FINE_TUNE_EPOCHS = 10     
FINE_TUNE_LAYERS = 30     


def split_data(paths, labels):
    """
    70% train  -> the model learns from these images
    15% val    -> checked after every epoch to watch for overfitting / early stopping
    15% test   -> never seen during training; used only once in evaluate.py
    stratify=labels keeps the same class proportions in every split.
    """
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED)
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def build_model(num_classes):
    """MobileNetV2 (pretrained on ImageNet) + a simple classification head."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False         # freeze: keep the pretrained feature extractor as it is

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    x = base_model(x, training=False)    # output: 7x7x1280 feature maps

    x = layers.GlobalAveragePooling2D()(x)

    # Output layer: one neuron per class. Softmax turns the scores into probabilities that sum to 1.
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",   # labels are plain integers 0..5
        metrics=["accuracy"])


def save_training_plot(history_list):
    """Join the history of both training stages and plot accuracy and loss."""
    acc, val_acc, loss, val_loss = [], [], [], []
    for history in history_list:
        acc += history.history["accuracy"]
        val_acc += history.history["val_accuracy"]
        loss += history.history["loss"]
        val_loss += history.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    head_epochs = len(history_list[0].history["accuracy"])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.axvline(head_epochs + 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.axvline(head_epochs + 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_history.png"), dpi=120)
    plt.close()


def main():
    tf.keras.utils.set_random_seed(SEED)              # makes results more repeatable
    for folder in (MODEL_DIR, PLOTS_DIR, RESULTS_DIR):
        os.makedirs(folder, exist_ok=True)

    # ---- 1. Load the list of images ----
    check_dataset()
    paths, labels = list_images()
    print(f"Found {len(paths)} images in {len(CLASS_NAMES)} classes.")

    # ---- 2. Split into train / validation / test ----
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_data(paths, labels)
    print(f"Train: {len(train_paths)}  Validation: {len(val_paths)}  Test: {len(test_paths)}")

    # Save the test split so evaluate.py uses exactly these unseen images.
    with open(TEST_SPLIT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(zip(test_paths, test_labels))

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels)

    # Class weights: the 'trash' class has far fewer images, so mistakes on it are
    # weighted more heavily. This is a simple way to handle class imbalance.
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=train_labels)
    class_weight = dict(enumerate(weights))
    print("Class weights:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weight.items()})

    # Early stopping: stop if validation loss hasn't improved for 3 epochs and
    # roll back to the best weights seen so far.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True)

    # ---- 3 & 4. Build the model and train only the new head ----
    model, base_model = build_model(len(CLASS_NAMES))
    compile_model(model, learning_rate=1e-3)
    model.summary()
    print("\nStage 1: training the classification head (MobileNetV2 frozen)...")
    history_head = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
                             class_weight=class_weight, callbacks=[early_stop])

    # ---- 5. Fine-tune the last layers of MobileNetV2 ----
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False          # keep the early (general) layers frozen
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False      # keep BatchNorm statistics fixed (standard fine-tuning practice)
    compile_model(model, learning_rate=1e-4)   # 10x smaller learning rate so we don't destroy pretrained weights
    print(f"\nStage 2: fine-tuning the last {FINE_TUNE_LAYERS} MobileNetV2 layers...")
    history_fine = model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
                             class_weight=class_weight, callbacks=[early_stop])

    # ---- 6. Save everything ----
    model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(CLASS_NAMES, f)
    save_training_plot([history_head, history_fine])

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc:.4f}")

    # Save a simple text log of every epoch (real numbers from this training run).
    with open(os.path.join(RESULTS_DIR, "training_log.txt"), "w") as f:
        f.write(f"Images: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}\n")
        for stage, history in (("Stage 1 (head)", history_head), ("Stage 2 (fine-tune)", history_fine)):
            f.write(f"\n{stage}\n")
            for i in range(len(history.history["loss"])):
                f.write(f"  epoch {i + 1}: "
                        f"loss={history.history['loss'][i]:.4f} "
                        f"acc={history.history['accuracy'][i]:.4f} "
                        f"val_loss={history.history['val_loss'][i]:.4f} "
                        f"val_acc={history.history['val_accuracy'][i]:.4f}\n")
        f.write(f"\nFinal validation accuracy (best weights): {val_acc:.4f}\n")
    print(f"Model saved to       {MODEL_PATH}")
    print(f"Class names saved to {CLASS_NAMES_PATH}")
    print(f"Test split saved to  {TEST_SPLIT_PATH}")
    print(f"Training plot saved to {os.path.join(PLOTS_DIR, 'training_history.png')}")
    print("Next step: python evaluate.py")


if __name__ == "__main__":
    main()
