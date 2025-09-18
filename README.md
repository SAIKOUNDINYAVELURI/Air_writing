# Air Writing Recognition ✍️

An AI-powered project that recognizes letters and words written in the air using hand-tracking and deep learning techniques. This system leverages computer vision to track hand gestures and predict corresponding text in real-time.

## 🚀 Features
- Real-time hand tracking using **MediaPipe/OpenCV**
- Recognizes air-written characters and words
- Deep Learning model for gesture-to-text prediction
- Live video feed with interactive UI
- Easy to extend with new datasets and models

## 🛠️ Tech Stack
- **Python 3.x**
- **OpenCV** – for video processing
- **MediaPipe** – for hand tracking
- **TensorFlow/Keras** – for deep learning model
- **NumPy & Pandas** – for preprocessing
- **Matplotlib/Seaborn** – for visualization

## 📂 Project Structure
```
Air_writing/
│── dataset/             # Training dataset (images/gestures)
│── models/              # Saved ML/DL models
│── src/                 # Main source code
│   ├── preprocessing.py # Data preprocessing scripts
│   ├── train.py         # Model training
│   ├── predict.py       # Prediction pipeline
│   ├── utils.py         # Helper functions
│── notebooks/           # Jupyter notebooks for experiments
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/SAIKOUNDINYAVELURI/Air_writing.git
   cd Air_writing
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate    # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
1. Run training (if you want to train a new model):
   ```bash
   python src/train.py
   ```

2. Run real-time air-writing recognition:
   ```bash
   python src/predict.py
   ```

3. (Optional) Preprocess dataset:
   ```bash
   python src/preprocessing.py
   ```

## 📊 Results
- Successfully recognizes characters written in air with hand gestures.

## 🔮 Future Improvements
- Add support for entire sentences instead of single characters.
- Optimize for speed on low-end devices.
- Integrate with AR/VR applications.

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repo, raise issues, or submit PRs.

---

