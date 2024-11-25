# PyTorchAudio: Audio Processing and Classification with PyTorch

Welcome to **PyTorchAudio**, a repository dedicated to audio processing and classification using PyTorch and Torchaudio. This project demonstrates how to preprocess audio data, build neural networks, and train models for sound classification tasks using datasets like UrbanSound8K.

---

## Repository Overview

This repository contains implementations for:
- **Audio preprocessing**:
  - Resampling, converting to mono, and extracting Mel Spectrograms.
- **Custom Dataset Class**:
  - Handling UrbanSound8K data with PyTorch `Dataset` and `DataLoader`.
- **Deep Learning Models**:
  - Building a Convolutional Neural Network (CNN) for audio classification.
- **Training and Evaluation**:
  - Training scripts with loss tracking, model saving, and prediction testing.

---

## Features

- **Efficient Data Loading**:
  - Use `urbandataset.py` to handle UrbanSound8K dataset.
- **Custom CNN Model**:
  - Designed to classify 10 sound classes effectively.
- **Audio Preprocessing**:
  - Powered by Torchaudio for Mel Spectrogram extraction and other transformations.
- **Training Script**:
  - A complete training pipeline with model saving and logging.
- **Prediction Script**:
  - Test the trained model on individual audio samples.

---

## Requirements

To run the project, ensure you have the following:
- Python 3.8+
- PyTorch 1.10+
- Torchaudio 0.10+
- NumPy, Pandas

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bahodir4/pytorchaudio.git
   cd pytorchaudio
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the UrbanSound8K dataset from [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), extract it, and place it in the project directory.

---

## Usage

### **1. Preprocessing and Training**
Use the `train.py` script to preprocess the dataset and train the CNN model:
```bash
python train.py
```
- Adjust the hyperparameters in the script as needed (e.g., `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`).

### **2. Making Predictions**
Use the `predict.py` script to test the trained model on individual audio samples:
```bash
python predict.py
```

### **3. Dataset Structure**
Ensure the dataset is structured as follows:
```
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── ...
├── metadata/
│   └── UrbanSound8K.csv
```

---

## File Structure

```plaintext
.
├── urbandataset.py          # Custom Dataset class for UrbanSound8K
├── cnn.py                   # CNN model implementation
├── train.py                 # Training script
├── predict.py               # Prediction script
├── requirements.txt         # Project dependencies
├── README.md                # Documentation
├── data/                    # Placeholder for dataset
└── models/                  # Directory to save trained models
```

---

## Model

### CNN Architecture
The implemented CNN consists of:
1. **Four convolutional layers**:
   - Extract spatial and temporal features from audio spectrograms.
2. **ReLU activations and MaxPooling**:
   - Enhance non-linearity and reduce dimensionality.
3. **Fully connected layers**:
   - Map features to 10 sound classes.
4. **Softmax output**:
   - Predict probabilities for the 10 classes.

---

## UrbanSound8K Dataset

The UrbanSound8K dataset contains urban sound recordings classified into the following 10 categories:
1. Air Conditioner
2. Car Horn
3. Children Playing
4. Dog Bark
5. Drilling
6. Engine Idling
7. Gun Shot
8. Jackhammer
9. Siren
10. Street Music

---

## Example Output

After running `predict.py`, you should see an output like this:
```
Predicted: drilling, Expected: drilling
```
This confirms the model's prediction matches the true label of the audio sample.

---

## Future Work

- **Augmentation**:
  - Add noise injection, time-stretching, or pitch-shifting for better generalization.
- **Advanced Models**:
  - Implement RNNs or Transformers for sequence modeling in audio tasks.
- **Live Prediction**:
  - Extend the system for real-time audio classification.

---

## Contributions

Contributions are welcome! Feel free to open an issue or submit a pull request to improve the project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- UrbanSound8K Dataset: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- PyTorch and Torchaudio: For providing powerful tools for deep learning and audio processing.

---

