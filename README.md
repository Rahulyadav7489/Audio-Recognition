
# Emotion Recognition from Speech

A deep learning project that uses LSTM neural networks to classify emotions from audio speech data. This project analyzes emotional speech patterns using the TESS (Toronto Emotional Speech Set) dataset and predicts seven different emotions.

## 📋 Project Overview

This notebook implements an end-to-end emotion recognition pipeline using:
- **Dataset**: TESS Toronto Emotional Speech Set (from Kaggle)
- **Audio Features**: MFCC (Mel-Frequency Cepstral Coefficients)
- **Model**: LSTM (Long Short-Term Memory) neural network
- **Emotions Detected**: Anger, Disgust, Fear, Happy, Neutral, Sad, Pleasant Surprise

## 🎯 Emotions Classified

The model can classify the following 7 emotions from speech:
1. **Angry** - Aggressive tone and delivery
2. **Disgust** - Repulsive or disapproving tone
3. **Fear** - Anxious or frightened delivery
4. **Happy** - Joyful and positive tone
5. **Neutral** - Emotionless, flat delivery
6. **Sad** - Sorrowful or melancholic tone
7. **Pleased/Surprised** - Pleasant surprise or delight

## 🛠️ Requirements

```
pandas
numpy
seaborn
matplotlib
librosa
librosa-display
scikit-learn
keras
tensorflow
IPython
```

## 📦 Installation

1. Clone or download this project
2. Install required packages:
   ```bash
   pip install pandas numpy seaborn matplotlib librosa scikit-learn keras tensorflow
   ```
3. Set up Kaggle API credentials:
   - Download `kaggle.json` from your Kaggle account
   - Store it in `~/.kaggle/kaggle.json`
   - Set appropriate permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 🚀 Usage

The notebook follows these main steps:

### 1. **Data Loading**
   - Downloads TESS dataset from Kaggle
   - Organizes audio files and corresponding emotion labels
   - Creates a structured DataFrame

### 2. **Exploratory Data Analysis (EDA)**
   - Visualizes distribution of emotions across dataset
   - Displays sample waveforms and spectrograms for each emotion
   - Plays audio samples for different emotional expressions

### 3. **Feature Extraction**
   - Extracts MFCC features from audio files
   - Uses 40 MFCC coefficients per audio sample
   - Creates feature vectors for model input

### 4. **Model Architecture**
   ```
   LSTM Layer (123 units) → Dense (64 units) → Dropout → 
   Dense (32 units) → Dropout → Dense (7 units, softmax)
   ```

### 5. **Model Training**
   - Trains on extracted MFCC features
   - 100 epochs with batch size of 512
   - 20% validation split
   - Categorical cross-entropy loss with Adam optimizer

### 6. **Results Visualization**
   - Training and validation accuracy curves
   - Training and validation loss curves

## 📊 Model Performance

The notebook generates performance visualizations:
- **Accuracy Plot**: Shows training vs validation accuracy over epochs
- **Loss Plot**: Shows training vs validation loss over epochs

## 🔍 Key Features

- **Audio Visualization**: Waveplot and spectrogram analysis for each emotion
- **MFCC Feature Extraction**: Captures temporal acoustic characteristics
- **Deep Learning**: LSTM captures sequential patterns in audio features
- **Dropout Regularization**: Prevents overfitting (20% dropout)
- **One-Hot Encoding**: Converts emotion labels for multi-class classification

## 📝 Project Structure

```
├── Emotion Recognition notebook
│   ├── Data Download (from Kaggle)
│   ├── Data Preprocessing
│   ├── EDA & Visualization
│   ├── Feature Extraction (MFCC)
│   ├── Model Building (LSTM)
│   ├── Model Training
│   └── Results Visualization
```

## 🎓 Learning Outcomes

This project demonstrates:
- Audio signal processing with librosa
- Deep learning with LSTM networks
- Feature engineering for audio data
- Model evaluation and visualization
- Working with Kaggle datasets

## 📚 Technologies Used

- **Python** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Librosa** - Audio processing
- **Keras/TensorFlow** - Deep learning framework
- **Scikit-learn** - Machine learning utilities
- **Matplotlib & Seaborn** - Data visualization

## 🔗 Dataset

**TESS (Toronto Emotional Speech Set)**
- Contains emotional speech recordings
- 7 different emotions
- Professional actors delivering scripted sentences
- Download from: https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess

## ⚙️ Hyperparameters

- **LSTM Units**: 123
- **Dense Layer 1**: 64 units (ReLU activation)
- **Dense Layer 2**: 32 units (ReLU activation)
- **Output Layer**: 7 units (Softmax activation)
- **Dropout Rate**: 0.2
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 100
- **Batch Size**: 512
- **Validation Split**: 0.2 (20%)

## 💡 Future Enhancements

- Test on real-world speech samples
- Implement data augmentation techniques
- Try alternative architectures (CNN, Transformer-based models)
- Deploy as a web API
- Real-time emotion detection from microphone input
- Multi-label emotion classification

## 📄 License

This project uses the TESS dataset. Please refer to the dataset's license terms.

## 👤 Author

Created as a deep learning project for emotion recognition from speech.

---

**Note**: Ensure you have a stable internet connection for downloading the Kaggle dataset and appropriate permissions for your Kaggle API credentials.
