## Overview

In this assignment, you will train a **Convolutional Neural Network (CNN)** using **PyTorch** on a subset of the QuickDraw dataset. You will be provided with a folder structured as follows:

```
selected_images/
    ambulance/
        ...
    angle/
        ...
    ...
```

Each subfolder represents a class label and contains all available images for that class (grayscale 28×28 PNG files).

Your objective is to design, train, evaluate, document, and report on a CNN capable of performing **multi-class image classification** on this dataset.

---

## Exercise — QuickDraw CNN Classifier

You must implement a complete deep-learning training and evaluation pipeline using PyTorch. You may choose any appropriate CNN architecture (custom, VGG-like, ResNet-like, etc.), as long as:

* All components are implemented in **PyTorch**
* Your notebook is fully reproducible
* Your model successfully trains on the dataset

### Dataset Description

You will receive data directory:

```
selected_images/
    ambulance/
        ...
    angle/
        ...
    ...
```

This dataset contains **100 classes**, each containing their images.

---

## Your Tasks

You must:

1. **Load the dataset** using a custom PyTorch `Dataset` and `DataLoader`.

2. **Split** the dataset into **training** and **test** sets (e.g., 80/20).

3. **Design and implement a CNN architecture** for 100-class classification.

4. **Train the model**, recording at each epoch:

   * Training loss
   * Training accuracy
   * Validation/Test loss
   * Validation/Test accuracy

5. **Plot** the following (required):

   * Training vs. test loss (per epoch)
   * Training vs. test accuracy (per epoch)

6. **Save your final trained model** as:

   ```
   model.pth
   ```

7. **Write reproducible evaluation code** that:

   * Loads `model.pth`
   * Computes test accuracy

8. **Prepare a PDF report** (details below).

You must submit a **PDF report** that includes:

### 1. CNN Architecture Description

* Clear description of the model architecture
* Number of layers, filter sizes, activation functions
* Explanation of design choices (why this architecture?)

### 2. Training Procedure

* Hyperparameters (learning rate, optimizer, batch size, etc.)
* Data preprocessing steps
* Train/test split strategy

### 3. Plots

Your report must contain the following plots produced by your code:

* **Training vs. test loss** over epochs
* **Training vs. test accuracy** over epochs

Plots must be generated inside your notebook and then inserted into the PDF.

### 4. Results & Discussion

* Final test accuracy
* Observations about underfitting/overfitting
* Strengths and weaknesses of your model
* Potential improvements

### PDF filename:

```
report.pdf
```

