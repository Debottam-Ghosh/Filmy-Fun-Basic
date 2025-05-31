# You Look Like... | Bollywood Celebrity Lookalike Detector

**"You Look Like..."** is a fun and engaging Streamlit app that uses facial recognition and deep learning to tell you **which Bollywood celebrity you resemble** most. Upload your image and get a match confidence score along with side-by-side photos!



---

## 🚀 Features

- 🧠 Face detection using **MTCNN**
- 🤖 Feature extraction with **Keras VGGFace (ResNet50)**
- 📊 Similarity scoring via **cosine similarity**
- 📈 Gauge chart visualization of match confidence
- 🖼️ Side-by-side comparison: **Your photo vs Celebrity**
- 🎨 Clean, themed UI with background image

---

## 📦 Tech Stack

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- [MTCNN](https://github.com/ipazc/mtcnn)
- [Keras VGGFace](https://github.com/rcmalli/keras-vggface)
- OpenCV, Pillow, NumPy, scikit-learn
- [Plotly](https://plotly.com/python/) for gauge chart

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Debottam-Ghosh/Filmy-Fun-Basic.git
cd Filmy-Fun-Basic
```

### 2. Create & Activate Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 📤 How to Use

1. Click "Choose your image"
2. Upload a clear, front-facing photo
3. Wait for the app to process
4. See your celebrity match and confidence score

---

## 🤝 Acknowledgments

- Keras VGGFace by rcmalli
- MTCNN Face Detector by ipazc
- Celebrity images and embeddings are used only for educational/demonstration purposes.

