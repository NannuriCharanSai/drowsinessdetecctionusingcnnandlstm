# 🚀  Drowsiness Detection

## 📖 Overview

This project is an **image classifier** designed to detect signs of **drowsiness** using two models: a **CNN** (Convolutional Neural Network) and a **CNN-LSTM** (Convolutional Long Short-Term Memory). The app classifies images into one of the following categories:

- 😴 **Yawn**: The person is yawning.
- 😐 **No Yawn**: The person is not yawning.
- 👀 **Open**: The person’s eyes are open.
- 😌 **Closed**: The person’s eyes are closed.

The GUI for this application is built using **Tkinter**.

---

## 📂 Project Structure

```plaintext
📁 train/
    ├── yawn/          # Images of yawning
    ├── no_yawn/       # Images of not yawning
    ├── Open/          # Images with open eyes
    └── Closed/        # Images with closed eyes
📁 preprocessed_sequences_*.npy    # Preprocessed image sequences for each category
📁 preprocessed_labels_*.npy       # Preprocessed labels for each category
📄 cnn_model.h5        # Trained CNN model
📄 drowsiness_detection_cnn_lstm_model.h5  # Trained CNN-LSTM model
📄 image_classifier.py  # Main code for the Tkinter GUI
📄 README.md            # Project README file (this file)
```

---

## ⚙️ Installation

To get started with this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-classifier-drowsiness.git
   ```

2. Install the dependencies using **pip**:
   ```bash
   pip install -r requirements.txt
   ```

   Dependencies include:
   - 🧠 **TensorFlow/Keras**: For building and running neural networks.
   - 📊 **Numpy**: For array manipulations.
   - 🖼️ **Pillow** (PIL): For image processing.
   - 🖱️ **Tkinter**: For GUI development (usually pre-installed with Python).

---

## 🛠️ Usage

1. **Preprocess Images**: Images are converted to grayscale, resized to **100x100 pixels**, and normalized to **[0, 1]** pixel values using:
   ```python
   img_array = np.array(img) / 255.0
   ```

2. **Running the Classifier**: Launch the app with:
   ```bash
   python image_classifier.py
   ```

3. **Using the GUI**:
   - Click the **"Select Image"** button to choose an image from your file system.
   - The app will display the image and provide predictions from both the **CNN** and **CNN-LSTM** models.

---

## 🧠 Model Architecture

### 📊 CNN Model
- **Input**: Grayscale image of size (100x100x1)
- **Architecture**:
  1. **Conv2D** with 32 filters (3x3 kernel) + ReLU activation
  2. **MaxPooling2D** (2x2 pool size)
  3. **Conv2D** with 64 filters (3x3 kernel) + ReLU activation
  4. **MaxPooling2D** (2x2 pool size)
  5. **Flatten** layer to convert 2D data into 1D
  6. **Dense** layer with 128 units + ReLU activation
  7. **Dropout** (50%) for regularization
  8. **Softmax** output layer (4 units, one for each class)

### 📈 CNN-LSTM Model
- Similar to the CNN model, but includes two **LSTM** layers after flattening the data, to capture temporal features (useful for sequential data).

---

## 📊 Training

- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: `Adam`
- **Metrics**: Accuracy
- **Training**:
   - Dataset: Preprocessed grayscale images of size **100x100 pixels**.
   - Data split: **80%** training and **20%** testing using `train_test_split()`.
   - Model trained for **10 epochs** with a batch size of **64**.

---

## 📈 Evaluation

- **Training Accuracy**: Evaluated on training data after training.
- **Testing Accuracy**: Evaluated on unseen test data using `model.evaluate()`.

Example output:
```plaintext
Train Accuracy: 98.75%
Test Accuracy: 95.00%
```

---

## 💻 GUI Details

- **Tkinter GUI**:
   - Displays the selected image in a **200x200 pixel** window.
   - Shows predictions from both the **CNN** and **CNN-LSTM** models.

### Key Functions:

1. **`preprocess_image()`**: 
   - Converts image to grayscale, resizes to **100x100**, and normalizes.
   - Reshapes the image for the model: `(100, 100, 1)`.

2. **`predict_cnn()`**:
   - Takes the preprocessed image and runs a prediction using the **CNN** model.
   - Returns the predicted class label.

3. **`predict_cnn_lstm()`**:
   - Runs prediction using the **CNN-LSTM** model.
   - Returns the predicted class label.

---

## 🚀 Future Enhancements

- 🕵️‍♂️ **Real-time Detection**: Implement real-time drowsiness detection from live video streams using the CNN-LSTM model.
- 💡 **Data Augmentation**: Improve generalization by adding more variations in lighting, angle, and noise to the dataset.
- 🎥 **Sequential Image Input**: Extend the CNN-LSTM model to handle video or continuous frame inputs for more accurate predictions.

