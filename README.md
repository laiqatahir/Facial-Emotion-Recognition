# Emotion Recognition using CNN (FER-2013)

A convolutional neural network that classifies human facial expressions into 7 emotion categories, trained on the FER-2013 dataset using TensorFlow/Keras.

## Overview

Facial emotion recognition is a challenging image processing problem due to subtle expressions, low image quality, and facial variation. This project builds a CNN from scratch to classify grayscale 48×48 facial images into one of 7 emotions using the public **FER-2013** dataset. CNNs are well suited to this task because they automatically learn visual features directly from pixel data through three stages: feature detection, feature reduction (pooling), and classification.

Data augmentation (rotation, shifting, zoom, horizontal flip, shear) is applied during training to improve generalization. For a task as difficult as FER-2013, **50–60% test accuracy is considered a reasonable result**.

## Model Architecture

- 3 convolutional blocks (64 → 128 → 256 filters) with `Conv2D`, `BatchNormalization`, `MaxPooling2D`, and `Dropout`
- `GlobalAveragePooling2D` followed by a dense layer (256 units) and dropout
- Final `Dense(7, softmax)` output layer for the 7 emotion classes
- Trained with the Adam optimizer, categorical cross-entropy loss, `EarlyStopping`, and `ReduceLROnPlateau` callbacks

## Repository Contents

| File | Description |
|---|---|
| `Facial_Expression_Recognition.ipynb` | Notebook covering data loading, preprocessing, model building, training, evaluation, and prediction |
| `Emotion-Recognition-using-CNN-FER2013.pptx` | Presentation covering the problem statement, CNN approach, dataset, and methodology |
| `emotion_recognition_cnn.h5` | Pre-trained model weights, ready for inference without retraining |
| Demo video | Video walkthrough of the notebook running end-to-end |

## Dataset

This project uses the **FER-2013** dataset, downloadable from Kaggle:
🔗 [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

It contains 48×48 grayscale facial images labeled with 7 emotion classes (angry, disgust, fear, happy, neutral, sad, surprise), split into `train` and `test` folders.

## Requirements

- Python 3
- `tensorflow`
- `opencv-python`
- `numpy`
- `matplotlib`

## How to Run

1. Download the FER-2013 dataset from Kaggle (link above) as a ZIP file.
2. Open `Facial_Expression_Recognition.ipynb` in [Google Colab](https://colab.research.google.com/) (or Jupyter, with the requirements above installed).
3. Run the cells in order:
   - When prompted, upload the FER-2013 ZIP file — it will be extracted automatically into `FER-2013/train` and `FER-2013/test`.
   - The notebook builds and compiles the CNN, then trains it for up to 20 epochs (with early stopping).
   - After training, the model is evaluated on the test set and the accuracy is printed.
4. **To predict on your own image**: run the final prediction cells and upload a facial image when prompted — the notebook will display the image along with its predicted emotion.

### Using the pre-trained model directly

If you don't want to retrain the model, load `emotion_recognition_cnn.h5` directly:

```python
from tensorflow.keras.models import load_model
model = load_model("emotion_recognition_cnn.h5")
```

Then preprocess an input image to 48×48 grayscale, normalize pixel values to [0, 1], reshape to `(1, 48, 48, 1)`, and call `model.predict(img)`.

## Limitations

- FER-2013 images are low resolution and often ambiguous, capping achievable accuracy at roughly 50–60% even for well-tuned models.
- The model only recognizes the 7 emotion classes present in FER-2013.
