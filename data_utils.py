import os

import tensorflow as tf

DATA_DIR = "dataset"          # folder that contains the 6 class folders
IMG_SIZE = 224                # MobileNetV2 was trained on 224x224 images
BATCH_SIZE = 32               # number of images processed together in one step
SEED = 42                     # fixed random seed so the split is the same every run

# The six garbage categories (must match the folder names inside dataset/)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Output locations
MODEL_DIR = "models"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"
MODEL_PATH = os.path.join(MODEL_DIR, "recyclevision_mobilenetv2.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
TEST_SPLIT_PATH = os.path.join(RESULTS_DIR, "test_split.csv")


def check_dataset():
    """Stop with a clear message if the dataset folders are missing."""
    for class_name in CLASS_NAMES:
        folder = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(folder):
            raise SystemExit(
                f"Dataset folder not found: {folder}\n"
                "Download the 6-class Garbage Classification dataset and put the folders\n"
                "cardboard, glass, metal, paper, plastic, trash directly inside 'dataset/'.\n"
                "See README.md -> 'Get the dataset'."
            )


def list_images():
    """
    Walk through dataset/<class_name>/ and collect every image path.
    Returns two lists of the same length:
      paths  -> file path of each image
      labels -> class number of each image (0 = cardboard, 1 = glass, ...)
    """
    paths = []
    labels = []
    for label, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(DATA_DIR, class_name)
        for file_name in sorted(os.listdir(folder)):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder, file_name))
                labels.append(label)
    return paths, labels


def load_image(path, label):
    """Read one image file from disk and resize it to 224x224."""
    image = tf.io.read_file(path)                                   # raw bytes
    image = tf.io.decode_image(image, channels=3,                    # bytes -> RGB pixel array
                               expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))             # every image becomes 224x224x3
    # Pixel values stay in the range 0-255 here. Normalisation to [-1, 1]
    # happens INSIDE the model (Rescaling layer), so the app can't forget it.
    return image, label


def make_dataset(paths, labels, shuffle=False):
    """Turn lists of paths/labels into a batched tf.data pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=SEED)  # mix the order each epoch
    dataset = dataset.map(load_image)                     # load + resize every image
    dataset = dataset.batch(BATCH_SIZE)                   # group images into batches of 32
    dataset = dataset.prefetch(tf.data.AUTOTUNE)          # prepare next batch while the model works
    return dataset
