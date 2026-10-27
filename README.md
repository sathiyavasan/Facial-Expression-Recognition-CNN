# Facial Expression Recognition Using CNN

An end-to-end Deep Learning implementation to perform Facial Expression Recognition (FER) using custom Convolutional Neural Networks built with Keras and TensorFlow.

## Project Architecture
The repository is organized into a clean, reusable structure:
- `dataset/`: Contains download guidelines and setup scripts for FER-2013.
- `notebooks/`: Holds training and verification notebooks.
- `models/`: Stores the trained CNN serialization files.
- `outputs/`: Diagnostic plots, class balance graphs, and classification reports.

## Seven Facial Expression Classes
The model evaluates grayscale human facial expressions and classifies them into seven categories:
1. **Angry**
2. **Disgust**
3. **Fear**
4. **Happy**
5. **Neutral**
6. **Sad**
7. **Surprise**

## Model Architecture
The custom Sequential Convolutional Neural Network uses four active block stages:
- **Convolutional Layer Blocks**: Utilizing 2D Convolutional layers (increasing dimensions from 32 up to 256 filters) coupled with Batch Normalization layers to accelerate learning and Dropout layers to combat overfitting.
- **Downsampling**: MaxPooling2D layers following each Conv block.
- **Classifier Head**: Global Average Pooling 2D layer coupled with a Dense layer (128 units, ReLU activation) and a final Dense classification layer (7 units, Softmax activation).

## Preprocessing & Pipeline
- Images are converted to single-channel grayscale and scaled to 48x48 resolutions.
- Normalization transforms raw pixel values from [0, 255] to a floating range of [0.0, 1.0].
- Real-time training augmentation is integrated using TensorFlow dataset layer transformations (random flips and rotations).

## Performance & Evaluation
- **Baseline Test Accuracy**: **58.68%**
- **Optimized Model Test Accuracy**: **61.77%**

Detailed training graphs, confusion matrix, and prediction logs can be reviewed in the `outputs/` folder.
