# RecycleVision – Viva Notes

These notes describe **exactly the code in this folder**. File and line references match the actual scripts.

---

## 1. Objective and problem statement

**Problem:** Sorting garbage by hand is slow, costly and error-prone. Recyclable items are often thrown in the wrong bin.

**Objective:** Build a deep learning model that looks at a photo of a waste item and classifies it into one of six
categories (**cardboard, glass, metal, paper, plastic, trash**), and deploy it through a simple Streamlit web app
that shows the predicted category and a confidence percentage.

**Use cases:** smart recycling bins, municipal waste sorting, teaching people how to segregate waste, and tracking
waste composition.

**Dataset:** TrashNet (six-class Garbage Classification dataset), 2,527 images:
cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137.
Each image shows one object on a plain background (original size 512×384).

---

## 2. Why deep learning for image classification?

- An image is just a big grid of numbers (224 × 224 × 3 = 150,528 values). Writing manual rules
  ("if shiny then metal") does not work because lighting, angle, and shape all vary.
- Deep learning **learns the features automatically** from examples: edges, then textures, then shapes, then objects.
- Convolutional neural networks are the standard, proven approach for images and reach high accuracy where
  classic machine learning on raw pixels does not.

## 3. Key concepts

**CNN (Convolutional Neural Network):** A neural network that slides small filters (for example 3×3) across the image.
Each filter detects a pattern such as an edge or a texture. Stacking many convolution layers lets the network detect
increasingly complex patterns. Pooling layers shrink the feature maps to keep the important information.

**MobileNetV2:** A CNN designed by Google to be **small and fast** (about 2.26 million parameters without its ImageNet output layer, versus about 23.5 million for ResNet50)
so it can run on phones and laptops. It uses *depthwise separable convolutions* and *inverted residual blocks*, which
need far fewer calculations than normal convolutions. It was pretrained on **ImageNet** (1.2 million images, 1,000 classes).

**Transfer learning:** Instead of training a CNN from scratch (which needs a huge dataset), we reuse a network that
already learned general visual features on ImageNet and only train a small new output layer for our six classes.
We have only about 1,770 training images, so transfer learning gives much better accuracy and trains in minutes.

**Freezing:** `base_model.trainable = False` means MobileNetV2's weights are not changed during training.
Only the new layer learns.

**Fine-tuning:** After the new layer has learned, we unfreeze the **last 30 layers** of MobileNetV2 and train them with
a learning rate 10 times smaller than stage 1 (0.0001), with the BatchNorm layers kept frozen, so they adapt slightly to garbage images without forgetting what they learned.

---

## 4. Files, in execution order

| Order | File | Purpose |
|---|---|---|
| – | `data_utils.py` | Shared settings (image size, paths, class names) and image-loading functions. Imported by the other scripts. |
| 1 | `eda.py` | Exploratory data analysis: class counts, sample images, average colour per class. |
| 2 | `train.py` | Splits data, builds the MobileNetV2 model, trains it, and saves the model, class names, test split and plots. |
| 3 | `evaluate.py` | Loads the saved model, predicts on the test set, and computes metrics and the confusion matrix. |
| 4 | `app.py` | Streamlit app: upload an image, see the prediction, confidence and top 3. |

### 4.1 `data_utils.py`

- **Settings block:** `DATA_DIR = "dataset"`, `IMG_SIZE = 224`, `BATCH_SIZE = 32`, `SEED = 42`, `CLASS_NAMES` (six names in
  alphabetical order), plus output paths (`models/`, `plots/`, `results/`).
- **`check_dataset()`:** stops with a helpful message if any of the six class folders is missing.
- **`list_images()`:** loops over each class folder and collects every `.jpg/.jpeg/.png` path. The label is the class's
  position in `CLASS_NAMES` (cardboard = 0, glass = 1, metal = 2, paper = 3, plastic = 4, trash = 5).
- **`load_image(path, label)`:**
  1. `tf.io.read_file` reads the raw bytes.
  2. `tf.io.decode_image(..., channels=3)` converts the bytes into an RGB pixel array.
  3. `tf.image.resize(image, (224, 224))` makes every image 224×224×3.
  Pixels stay in the range 0–255 here. Normalisation is done inside the model (see the Rescaling layer).
- **`make_dataset(paths, labels, shuffle)`:** builds a `tf.data` pipeline that optionally shuffles, then
  `map(load_image)`, `batch(32)`, and `prefetch` (loads the next batch while the model is busy).

### 4.2 `eda.py`

1. Counts images per class, prints them, and saves `plots/class_distribution.png`. This shows that **trash is much smaller
   (137 images)**, so there is class imbalance.
2. Saves a grid of four example images per class to `plots/sample_images.png`.
3. Computes the average R, G, B value of up to 50 images per class and saves `plots/color_distribution.png`.
   **Finding:** all classes have similar, bright average colours (about 150–180) because of the plain light backgrounds.
   Only cardboard stands out, with clearly less blue, because it is brown. So colour alone cannot separate the classes,
   and the model must learn shapes and textures. That is exactly what a CNN does.

### 4.3 `train.py`, block by block

1. **Settings:** `HEAD_EPOCHS = 10`, `FINE_TUNE_EPOCHS = 10`, `FINE_TUNE_LAYERS = 30`.
2. **`main()` start:** `set_random_seed(42)` for repeatability, and creates the `models/`, `plots/` and `results/` folders.
3. **Load the list of images:** `check_dataset()` then `list_images()`.
4. **`split_data()`:** two calls to scikit-learn's `train_test_split`:
   - First: 70% train and 30% temporary.
   - Second: the 30% is split in half, giving 15% validation and 15% test.
   - `stratify=labels` keeps the same class proportions in each split, and `random_state=42` makes the split identical every run.
   - Roughly 1,768 train, 379 validation and 380 test images.
5. **Save test split:** writes `results/test_split.csv` (path and label) so `evaluate.py` uses **exactly** the unseen test images.
6. **Datasets:** `make_dataset(train..., shuffle=True)` and `make_dataset(val...)`.
7. **Class weights:** `compute_class_weight("balanced", ...)` gives rare classes (trash) a higher weight in the loss,
   so the model cannot ignore them. Formula: `weight = total_images / (num_classes × images_in_class)`.
8. **Early stopping:** watches `val_loss`. If it hasn't improved for three epochs, training stops, and
   `restore_best_weights=True` rolls back to the best epoch.
9. **`build_model()`**, layer by layer:
   - `MobileNetV2(include_top=False, weights="imagenet")`: loads the pretrained network **without** its original 1,000-class
     output layer. `base_model.trainable = False` freezes it.
   - `Input(shape=(224, 224, 3))`.
   - `RandomFlip("horizontal")`, `RandomRotation(0.1)` (±10% of a full turn, which is up to ±36°), and `RandomZoom(0.1)`:
     **data augmentation**. Each epoch the model sees slightly different versions of the images, which reduces overfitting.
     These layers are only active during training and do nothing during prediction.
   - `Rescaling(1/127.5, offset=-1)`: **normalisation** from 0..255 to -1..1, which is the range MobileNetV2 was trained with
     (0 becomes -1, 127.5 becomes 0, 255 becomes 1). Because it is inside the model, the app never forgets to normalise.
   - `base_model(x, training=False)`: runs MobileNetV2 and outputs a 7×7×1280 block of features. `training=False` asks
     MobileNetV2 to run its BatchNorm layers in inference mode. In stage 2 we also set the BatchNorm layers to
     `trainable = False` to make sure of this.
   - `GlobalAveragePooling2D()`: averages each of the 1,280 7×7 maps into a single number, giving a vector of 1,280 features.
   - `Dense(6, activation="softmax")`: **the output layer**. It has one neuron per class, and softmax turns the six scores
     into probabilities. It has 1,280 × 6 weights + 6 biases = **7,686 parameters**.
   - From the real `model.summary()` output: **total 2,265,670 parameters**, of which only **7,686 are trainable in stage 1**.
     The other 2,257,984 belong to the frozen MobileNetV2.
10. **`compile_model()`:** Adam optimizer and `sparse_categorical_crossentropy` loss (labels are integers 0–5), with
    accuracy as the metric.
11. **Stage 1 (`model.fit`, learning rate 0.001):** only the Dense layer learns, because MobileNetV2 is frozen.
12. **Stage 2 (fine-tuning):** `base_model.trainable = True`, then the layers before the last 30 are frozen again. The model is
    re-compiled with learning rate 0.0001 (10× smaller than stage 1) and trained for up to 10 more epochs. **All BatchNormalization layers are also kept frozen** (`layer.trainable = False`). Otherwise they would update their mean/variance statistics on our small batches and disturb the pretrained features. (Our first training run without this line showed training accuracy suddenly dropping from 0.90 to 0.77 when fine-tuning began.)
13. **Save:** `model.save("models/recyclevision_mobilenetv2.keras")`, `class_names.json`,
    `plots/training_history.png` (accuracy and loss curves with a dashed line where fine-tuning starts), and
    `results/training_log.txt` (the numbers for every epoch).

### 4.4 `evaluate.py`

1. Loads the saved model, `class_names.json`, and `results/test_split.csv`.
2. `make_dataset(test...)` without shuffling, so the prediction order matches `true_labels`.
3. `model.predict(test_ds)` returns a (380, 6) array of probabilities. `np.argmax(..., axis=1)` picks the class with the
   highest probability for each image.
4. Metrics from scikit-learn: `accuracy_score`, `precision_recall_fscore_support(average="macro")`, and
   `classification_report` (per class).
5. `confusion_matrix` and `ConfusionMatrixDisplay` are saved to `plots/confusion_matrix.png`. Text results are saved to
   `results/evaluation_report.txt`.

### 4.5 `app.py`

1. `st.set_page_config` sets the page title and icon.
2. `@st.cache_resource load_model()` loads the `.keras` model and class names **once**, not every time the page reruns.
3. If the model file doesn't exist, the app shows an error and `st.stop()`.
4. `st.file_uploader` accepts a jpg, jpeg or png image.
5. **`predict()`:** the image is converted to RGB, then a NumPy array, then `tf.image.resize` to 224×224 (the same as training),
   then `expand_dims` gives shape (1, 224, 224, 3), because models always take a batch. Then `model.predict` runs.
   Augmentation is off and Rescaling is applied automatically inside the model.
6. `np.argmax` gives the best class, and `probabilities[best] * 100` gives the confidence %.
7. The app shows the predicted category (`st.success`), the confidence (`st.metric`), and the top 3 classes with progress bars.

---

## 5. How an image travels through the system

```
JPG file → read bytes → decode to RGB (H×W×3, 0–255) → resize to 224×224×3
        → [training only: random flip/rotate/zoom]
        → Rescaling to [-1, 1]
        → MobileNetV2 (frozen/fine-tuned) → 7×7×1280 features
        → GlobalAveragePooling → 1280 numbers
        → Dense(6) + softmax → 6 probabilities, e.g. [0.02, 0.05, 0.85, 0.03, 0.04, 0.01]
        → argmax → "metal", confidence 85%
```

## 6. Output layer, softmax and confidence

- The last Dense layer produces six raw scores (logits), one per class.
- **Softmax** converts them into probabilities: `p_i = e^(score_i) / Σ e^(score_j)`. All values are between 0 and 1 and
  **add up to 1**.
- The **predicted class** is the one with the highest probability (argmax).
- **Confidence %** = that highest probability × 100. A confidence of 92% means the model assigns 92% of its probability to
  that class. It does **not** guarantee the prediction is correct, and models can be confidently wrong.

---

## 7. Evaluation metrics, explained simply

Example for the class "glass":
- **TP (true positive):** a glass image predicted as glass.
- **FP (false positive):** a non-glass image predicted as glass.
- **FN (false negative):** a glass image predicted as something else.

| Metric | Formula | Meaning in plain words |
|---|---|---|
| **Accuracy** | correct / total | What fraction of all test images were classified correctly. |
| **Precision** | TP / (TP + FP) | "When the model says *glass*, how often is it really glass?" |
| **Recall** | TP / (TP + FN) | "Of all real glass images, how many did the model find?" |
| **F1-score** | 2 × P × R / (P + R) | Harmonic mean of precision and recall. It is high only if **both** are high. |

- **Macro average:** compute the metric separately for each class, then take the plain average. Every class counts equally,
  so poor performance on the small *trash* class is not hidden. (Weighted average gives bigger classes more weight.)
- **Confusion matrix:** a 6×6 table where **rows are the true class and columns are the predicted class**.
  The diagonal shows correct predictions, and off-diagonal cells show which classes get confused
  (for example glass vs plastic, or paper vs cardboard).
- Why not accuracy alone? With imbalanced classes, a model could ignore *trash* and still have good accuracy.
  Precision, recall and F1 per class reveal that problem.

---

## 8. Our actual results (real numbers from this project)

All numbers come from `results/training_log.txt`, `results/evaluation_report.txt` and `plots/confusion_matrix.png`
(run on 17 Oct 2026, CPU only, TensorFlow 2.21, about 13 minutes of training).

**Split:** 1,768 train, 379 validation and 380 test images.

**Training (from `training_log.txt`):**
- Stage 1 (frozen base, 10 epochs): validation accuracy rose from 72.8% to a best of 82.9%.
- Stage 2 (fine-tuning): validation accuracy reached 87.6% at stage-2 epoch 6 (lowest validation loss, 0.404).
  Early stopping halted training at epoch 9 and **restored the epoch-6 weights**.
- Training accuracy ended near 98% while validation stayed about 87%, which shows **some overfitting**
  (the gap between the blue and orange lines in `training_history.png`).

**Test set (380 unseen images, evaluated once):**

| Metric | Score |
|---|---|
| Accuracy | **86.58%** (329 of 380 correct) |
| Precision (macro) | 87.35% |
| Recall (macro) | 85.37% |
| F1-score (macro) | 85.84% |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| cardboard | 0.914 | 0.883 | 0.898 |
| glass | 0.949 | 0.737 | 0.830 |
| metal | 0.769 | 0.968 | 0.857 |
| paper | 0.863 | 0.921 | 0.891 |
| plastic | 0.863 | 0.863 | 0.863 |
| trash | 0.882 | 0.750 | 0.811 |

**Reading the confusion matrix:**
- **Glass has the lowest recall (73.7%).** Of 76 glass images, 10 were predicted as metal and 8 as plastic.
  Shiny glass looks like metal, and clear glass looks like clear plastic bottles.
- **Metal has the lowest precision (76.9%).** The model says "metal" too easily, mostly for glass (10) and plastic (6).
- **Cardboard and paper get mixed up:** 7 cardboard images were predicted as paper, and 4 paper images as cardboard.
  Both are flat, fibrous and printed.
- **Trash:** 15 of 20 were correct (75% recall). It is the smallest and most varied class.
- Best classes: metal recall 96.8%, paper recall 92.1%, cardboard F1 0.898.

**What changed during development (be honest if asked):**
our first training run fine-tuned with learning rate 1e-5 and without freezing BatchNorm. It only reached
82.1% validation accuracy, and training accuracy dropped from 0.90 to 0.77 when fine-tuning started.
We then froze the BatchNorm layers, used learning rate 1e-4, and allowed up to 10 fine-tuning epochs, which gave 87.6%.
This decision was made using **validation** accuracy only. The test set was evaluated only once, on the final model.

---

## 9. Likely viva questions and answers

1. **What is the aim of your project?** To classify a waste image into six categories using a deep learning model and
   show the result in a Streamlit app, to help automate recycling.
2. **Which dataset did you use?** TrashNet (six-class Garbage Classification), 2,527 images: cardboard, glass, metal, paper,
   plastic and trash.
3. **Why resize to 224×224?** MobileNetV2 was pretrained on 224×224 images, and a neural network needs every input to be
   the same size.
4. **How did you normalise pixels?** With a `Rescaling(1/127.5, offset=-1)` layer inside the model, which maps 0–255 to -1..1,
   the range MobileNetV2 expects.
5. **How did you split the data?** Stratified 70% train, 15% validation and 15% test with seed 42, using `train_test_split`
   twice.
6. **Why do you need a validation set and a test set?** Validation is checked after every epoch to tune and stop training
   (early stopping). Test is used only once at the end, so it gives an honest estimate on unseen data.
7. **What is transfer learning?** Reusing a model trained on a large dataset (ImageNet) as a feature extractor and training
   only a new output layer for our task.
8. **Why MobileNetV2 and not ResNet50?** It is small (about 2.26 million parameters in the base, versus about 23.5 million for ResNet50) and fast, so it suits a web app and edge
   devices like smart bins, and it still gives good accuracy.
9. **What does freezing mean?** Setting `trainable = False`, so the pretrained weights do not change during training.
10. **What is fine-tuning and why a small learning rate?** We unfreeze the last 30 MobileNetV2 layers and train them gently.
    A 10× smaller rate (1e-4) adapts them to garbage images without destroying the pretrained knowledge.
11. **What does GlobalAveragePooling2D do?** It averages each 7×7 feature map into one value, turning 7×7×1280 into a vector
    of 1,280 numbers, with far fewer parameters than Flatten.
12. **What is softmax?** An activation that turns raw scores into probabilities between 0 and 1 that add up to 1.
13. **Which loss function did you use and why?** Sparse categorical cross-entropy, because this is multi-class classification
    and the labels are integers (0–5) rather than one-hot vectors.
14. **What optimizer?** Adam, an adaptive gradient-descent optimizer that works well with little tuning. The learning rate
    is 0.001 in stage 1 and 0.0001 in stage 2.
15. **What is data augmentation and which ones did you use?** Creating modified copies of the training images (horizontal
    flip, rotation up to ±36°, and zoom up to 10%) to reduce overfitting. It is active only during training.
16. **How did you handle class imbalance?** Class weights (`compute_class_weight("balanced")`), so mistakes on the rare
    *trash* class cost more, plus stratified splitting.
17. **What is early stopping?** It stops training when validation loss hasn't improved for three epochs and restores the
    best weights. This prevents overfitting and saves time.
18. **What is overfitting and how do you detect it?** The model memorises the training data and does poorly on new data.
    In the plot, training accuracy keeps rising while validation accuracy stops improving or drops.
19. **What do precision and recall mean?** Precision: of the images predicted as class X, how many really are X.
    Recall: of the real X images, how many were found.
20. **Why F1-score?** It balances precision and recall in one number, which is useful when classes are imbalanced.
21. **How do you read the confusion matrix?** Rows are the true class and columns are the predicted class. The diagonal is
    correct, and off-diagonal cells show which classes get confused.
22. **What does the confidence percentage mean?** The softmax probability of the predicted class × 100. It shows how sure
    the model is, but it is not a guarantee.
23. **What is an epoch and a batch?** An epoch is one full pass over the training images. A batch is the 32 images processed
    together before each weight update.
24. **Why is the model saved in `.keras` format?** It is the native Keras 3 format and stores the architecture, weights and
    the preprocessing layers together in one file.
25. **Why `@st.cache_resource` in the app?** Streamlit reruns the whole script on every interaction, and caching loads the
    model only once, so predictions are fast.
26. **What accuracy did your model get?** 86.58% on 380 unseen test images, with macro F1 85.84% and
    best validation accuracy 87.6%. This is above the 85% target in the project brief.
27. **Which classes does the model confuse most, and why?** Glass is predicted as metal (10 cases) or plastic (8),
    because they are shiny or transparent. Cardboard is predicted as paper (7), because both are flat printed fibre.
28. **Is your model overfitting?** Slightly. Training accuracy reached about 98% while validation stayed about 87%.
    Augmentation, early stopping (restoring the best epoch) and the frozen base help limit it.
29. **Why is the test set used only once?** If we changed the model after looking at test results, the test score
    would no longer be an honest estimate of performance on new data. We tuned using the validation set only.
30. **Why does the app sometimes show 100.0% confidence?** The probability is rounded to one decimal place, so for
    example 99.97% is displayed as 100.0%. The model is very sure, but that is not a guarantee of correctness.

---

## 10. Limitations

- Small dataset (2,527 images), with only 137 *trash* images.
- Images have plain backgrounds with a single object, so real-world photos (cluttered bins, many objects, dirty items)
  may be classified less accurately.
- Only six classes. There is no *organic/biological* class, although the project brief mentions it, because TrashNet
  doesn't include it.
- Visually similar classes (glass vs plastic, paper vs cardboard) are naturally confused.
- Only one model (MobileNetV2) was trained. No comparison with a baseline CNN, ResNet50 or EfficientNet.
- Confidence is not calibrated: the model can be confidently wrong, and it always picks one of the six classes even for
  a non-garbage photo.
- Some overfitting (about 98% train accuracy vs about 87% validation accuracy).
- Glass recall is only 73.7%, and metal precision is 76.9% (glass is often predicted as metal or plastic).
- Trained on CPU with few epochs, so results could improve with more training.

## 11. Future improvements

- Use the larger 12-class Kaggle dataset (about 15,000 images), which includes biological, batteries and clothes.
- Compare with a baseline CNN and with EfficientNetB0 or ResNet50, and pick the model with the best F1-score.
- Add a confidence threshold ("not sure, please check") and an "unknown/not garbage" option.
- Object detection (for example YOLO) to find several items in one photo.
- Convert to TensorFlow Lite to run on a Raspberry Pi inside a smart bin.
- Deploy on Streamlit Community Cloud so anyone can use it with a link.
