import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_utils import CLASS_NAMES, PLOTS_DIR, check_dataset, list_images


def main():
    check_dataset()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    paths, labels = list_images()

    # ---- 1. Number of images per class ----
    counts = [labels.count(i) for i in range(len(CLASS_NAMES))]
    for name, count in zip(CLASS_NAMES, counts):
        print(f"{name:10s}: {count}")
    print(f"{'total':10s}: {len(paths)}")

    plt.figure(figsize=(8, 4))
    bars = plt.bar(CLASS_NAMES, counts, color="seagreen")
    plt.bar_label(bars)
    plt.title("Number of images per class")
    plt.ylabel("Images")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=120)
    plt.close()

    # ---- 2. Example images: 4 per class ----
    per_class = 4
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(per_class * 2.2, len(CLASS_NAMES) * 2.2))
    for row, class_name in enumerate(CLASS_NAMES):
        class_paths = [p for p, l in zip(paths, labels) if l == row][:per_class]
        for col, path in enumerate(class_paths):
            axes[row, col].imshow(Image.open(path).convert("RGB"))
            axes[row, col].axis("off")
        axes[row, 0].set_title(class_name, loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sample_images.png"), dpi=100)
    plt.close()

    # ---- 3. Average colour per class (from up to 50 images, resized small for speed) ----
    mean_colours = []
    for row, class_name in enumerate(CLASS_NAMES):
        class_paths = [p for p, l in zip(paths, labels) if l == row][:50]
        pixels = [np.array(Image.open(p).convert("RGB").resize((64, 64))) for p in class_paths]
        mean_colours.append(np.mean(pixels, axis=(0, 1, 2)))   # mean R, G, B
    mean_colours = np.array(mean_colours)

    x = np.arange(len(CLASS_NAMES))
    plt.figure(figsize=(9, 4))
    for i, (channel, colour) in enumerate(zip(["Red", "Green", "Blue"], ["red", "green", "blue"])):
        plt.bar(x + (i - 1) * 0.25, mean_colours[:, i], width=0.25, label=channel, color=colour, alpha=0.7)
    plt.xticks(x, CLASS_NAMES)
    plt.ylabel("Mean pixel value (0-255)")
    plt.title("Average colour per class")
    plt.ylim(0, 255)
    plt.legend(loc="upper center", ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "color_distribution.png"), dpi=120)
    plt.close()

    print(f"EDA plots saved in '{PLOTS_DIR}/'")


if __name__ == "__main__":
    main()
