##  Color Predictor with LSTM Networks

This project implements a **deep learning model** using **LSTM (Long Short-Term Memory) networks** to predict **RGB color values** and visualize the color from their names. The model learns the relationship between textual color descriptions and their corresponding RGB representations using **TensorFlow**.

---

##  Project Overview

Color naming is a fascinating intersection of **linguistics and perception**. This project tackles the challenge of mapping natural language color descriptions to precise RGB values using deep learning.

**Key Features:**
- **End-to-end pipeline** from text preprocessing to RGB prediction
- **Sequence-to-vector mapping** using stacked LSTM architecture
- **Real-time color visualization** of predictions
- **Comprehensive training analysis** with accuracy and loss tracking

---

##  Dataset

The model is trained on a CSV dataset containing color names with their corresponding RGB values:

| Column | Description | Range |
|--------|-------------|-------|
| `name` | Color name (e.g., "parakeet", "pool blue") | Variable length strings |
| `red` | Red channel value | 0-255 |
| `green` | Green channel value | 0-255 |
| `blue` | Blue channel value | 0-255 |

**Dataset Statistics:**
- **Color name length distribution**: Normal distribution analysis performed
- **Maximum name length**: 30 characters (truncated to 25 for efficiency)
- **Character vocabulary**: 28 unique characters including letters and spaces
- **RGB normalization**: Values scaled from [0-255] to [0-1] for training stability

---

##  Model Architecture

### Network Design
```python
Sequential([
    LSTM(256, return_sequences=True, input_shape=(25, 28)),  # First LSTM layer
    LSTM(128),                                                # Second LSTM layer  
    Dense(128, activation='relu'),                           # Fully connected layer
    Dense(3, activation='sigmoid')                           # RGB output layer
])
```

### Architecture Details
- **Input**: One-hot encoded character sequences (25 timesteps × 28 characters)
- **LSTM Layers**: Hierarchical feature extraction with 256→128 units
- **Dense Layers**: 128-unit ReLU layer for feature combination
- **Output**: 3-unit sigmoid layer for normalized RGB values

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Accuracy
- **Epochs**: 40
- **Batch Size**: 32
- **Validation Split**: 10%

---

##  Performance Results

The model demonstrates strong learning capabilities with consistent improvement over 40 epochs:

- **Final Training Accuracy**: ~83.7%
- **Final Validation Accuracy**: ~70.4%
- **Training Loss**: 0.0055 (MSE)
- **Validation Loss**: 0.0481 (MSE)

### Training Characteristics
- **Convergence**: Steady improvement with minimal overfitting
- **Stability**: Consistent validation performance after epoch 20
- **Generalization**: Good balance between training and validation metrics

---

##  Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Zeyad-Baloch/colors-prediction-by-name-using-LSTM.git
cd colors-prediction-by-name-using-LSTM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.19.0
numpy
pandas
matplotlib
scipy
keras
```

### 3. Prepare Dataset
Ensure your `colors.csv` file is in the project directory with the following format:
```csv
name,red,green,blue
parakeet,174,182,87
saddle brown,88,52,1
cucumber crush,222,237,215
```

---

##  Usage

### Training the Model
```python
python color_predictor.py
```

### Making Predictions
```python
# Predict RGB values for a color name
predict("sunset orange")
predict("ocean blue")
predict("forest green")
```

### Custom Predictions
```python
def predict_custom(color_name):
    tokenized = t.texts_to_sequences([color_name.lower()])
    padded = preprocessing.sequence.pad_sequences(tokenized, maxlen=25)
    one_hot = to_categorical(padded, num_classes=28)
    prediction = model.predict(np.array(one_hot))[0]
    rgb = [int(p * 255) for p in prediction]
    return rgb
```

---

##  Technical Implementation

### Text Processing Pipeline
1. **Character-level tokenization** with Keras Tokenizer
2. **Sequence padding** to fixed length (25 characters)
3. **One-hot encoding** for neural network input
4. **Vocabulary mapping** of 28 unique characters

### Data Preprocessing
1. **RGB normalization** to [0-1] range using `value/255.0`
2. **Statistical analysis** of name length distribution
3. **Sequence length optimization** based on dataset characteristics

### Model Training
1. **Validation split** for performance monitoring
2. **Epoch-wise tracking** of accuracy and loss metrics
3. **Training visualization** with matplotlib graphs

---


##  Contributing

Contributions are welcome! Here are some ways to contribute:

- **Model improvements**: Experiment with different architectures
- **Visualization enhancements**: Create better prediction displays
- **Performance optimization**: Improve training efficiency
- **Documentation**: Enhance code comments and examples

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**⭐ If you find this project helpful, please consider giving it a star!**
