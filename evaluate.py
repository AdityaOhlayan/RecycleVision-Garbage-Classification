import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from data_utils import CLASS_NAMES_PATH, MODEL_PATH, PLOTS_DIR, RESULTS_DIR, TEST_SPLIT_PATH, make_dataset


def read_test_split():
    """Read the test image paths and labels saved by train.py."""
    paths, labels = [], []
    with open(TEST_SPLIT_PATH, newline="") as f:
        for row in csv.DictReader(f):
            paths.append(row["path"])
            labels.append(int(row["label"]))
    return paths, labels


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_SPLIT_PATH):
        raise SystemExit("Trained model or test split not found. Run 'python train.py' first.")

    # ---- 1. Load the trained model, class names and test images ----
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    test_paths, true_labels = read_test_split()
    test_ds = make_dataset(test_paths, true_labels)      # no shuffling: order must match true_labels
    print(f"Evaluating on {len(test_paths)} test images...")

    # ---- 2. Predict ----
    probabilities = model.predict(test_ds)               # shape: (num_images, 6), each row sums to 1
    predicted_labels = np.argmax(probabilities, axis=1)  # the class with the highest probability

    # ---- 3. Metrics ----
    accuracy = accuracy_score(true_labels, predicted_labels)
    # "macro" = compute the metric for each class, then take the simple average,
    # so every class counts equally (even the small 'trash' class).
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro")
    report = classification_report(true_labels, predicted_labels,
                                   target_names=class_names, digits=4)

    summary = (
        "RecycleVision - Test set results\n"
        f"Test images:        {len(test_paths)}\n"
        f"Accuracy:           {accuracy:.4f}\n"
        f"Precision (macro):  {precision:.4f}\n"
        f"Recall (macro):     {recall:.4f}\n"
        f"F1-score (macro):   {f1:.4f}\n\n"
        "Per-class report:\n" + report
    )
    print(summary)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
        f.write(summary)

    # ---- 4. Confusion matrix ----
    # Rows = true class, columns = predicted class. The diagonal = correct predictions.
    matrix = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (test set)")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    print(f"Report saved to {os.path.join(RESULTS_DIR, 'evaluation_report.txt')}")
    print(f"Confusion matrix saved to {os.path.join(PLOTS_DIR, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()
