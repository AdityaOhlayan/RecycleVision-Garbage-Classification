# ♻️ RecycleVision – Garbage Image Classification Using Deep Learning

RecycleVision classifies a photo of waste into one of six categories:
**cardboard, glass, metal, paper, plastic, trash**.
It uses **transfer learning with MobileNetV2** (pretrained on ImageNet) in TensorFlow/Keras,
and a **Streamlit** web app for uploading an image and seeing the prediction with a confidence percentage.

## Project structure

```
Mini-Project-4/
├── data_utils.py        # shared settings + image loading helpers
├── eda.py               # exploratory data analysis plots
├── train.py             # split data, build & train MobileNetV2 model, save model + plots
├── evaluate.py          # accuracy, precision, recall, F1, confusion matrix on the test set
├── app.py               # Streamlit app (upload image -> prediction + confidence)
├── requirements.txt
├── VIVA_NOTES.md        # explanation of the code and likely viva questions
├── dataset/             # the images (not committed to git)
│   ├── cardboard/  glass/  metal/  paper/  plastic/  trash/
├── models/              # created by train.py
│   ├── recyclevision_mobilenetv2.keras
│   └── class_names.json
├── plots/               # created by eda.py, train.py, evaluate.py
│   ├── class_distribution.png  sample_images.png  color_distribution.png
│   ├── training_history.png    confusion_matrix.png
└── results/             # created by train.py, evaluate.py
    ├── test_split.csv
    ├── training_log.txt
    └── evaluation_report.txt
```

## Get the dataset

This project uses the **TrashNet** dataset (the six-class "Garbage Classification" dataset, 2,527 images).

1. Download `dataset-resized.zip` from <https://huggingface.co/datasets/garythung/trashnet>
   (or the Kaggle "Garbage classification" 6-class dataset).
2. Extract it so the six class folders sit **directly** inside `dataset/`:
   `dataset/cardboard`, `dataset/glass`, `dataset/metal`, `dataset/paper`, `dataset/plastic`, `dataset/trash`.
   (If the zip gives you a nested folder like `dataset-resized/` or `Garbage classification/Garbage classification/`,
   move the six inner folders up into `dataset/`.)

## Setup (Windows)

TensorFlow needs **Python 3.9–3.12** (it does not support 3.14 yet).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the project (in this order)

```bash
python eda.py          # 1. dataset plots -> plots/
python train.py        # 2. train the model -> models/, plots/training_history.png, results/
python evaluate.py     # 3. test-set metrics -> results/evaluation_report.txt, plots/confusion_matrix.png
streamlit run app.py   # 4. open the web app in the browser
```

Training runs on the CPU and takes roughly 15–40 minutes depending on your laptop.
If training is interrupted, just run `python train.py` again. It starts from the beginning and
overwrites the previous model files.

## How it works (short version)

1. **Load and resize:** every image is read and resized to 224×224×3.
2. **Split:** stratified 70% train, 15% validation, 15% test (fixed random seed 42).
3. **Model:** augmentation (flip/rotate/zoom), then Rescaling to [-1, 1], frozen MobileNetV2,
   GlobalAveragePooling2D, then Dense(6, softmax).
4. **Training:** stage 1 trains only the new Dense layer, and stage 2 fine-tunes the last 30 MobileNetV2 layers
   with a 10× smaller learning rate (BatchNorm layers stay frozen). Early stopping and class weights are used.
5. **Evaluation:** genuine predictions on the 15% test set that was never used during training.
6. **App:** upload an image, the same resize is applied, the model gives 6 probabilities, and the highest one
   is shown as the prediction and its confidence.

## Results (actual run, 17 Oct 2026, CPU, TensorFlow 2.21)

Test set: **380 images** never used in training (`results/evaluation_report.txt`).

| Metric | Score |
|---|---|
| Accuracy | **86.58%** |
| Precision (macro) | 87.35% |
| Recall (macro) | 85.37% |
| F1-score (macro) | 85.84% |

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| cardboard | 0.914 | 0.883 | 0.898 | 60 |
| glass | 0.949 | 0.737 | 0.830 | 76 |
| metal | 0.769 | 0.968 | 0.857 | 62 |
| paper | 0.863 | 0.921 | 0.891 | 89 |
| plastic | 0.863 | 0.863 | 0.863 | 73 |
| trash | 0.882 | 0.750 | 0.811 | 20 |

Best validation accuracy was 87.6%. Training took about 13 minutes on CPU (about 40 s per epoch).
The model file is 21 MB (`models/recyclevision_mobilenetv2.keras`).
Plots: `plots/training_history.png` and `plots/confusion_matrix.png`.

If you re-run `train.py`, numbers can differ slightly because CPU floating-point operations are not perfectly deterministic.
