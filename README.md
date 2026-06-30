# 🎬 IMDB Movie Review Sentiment Analysis

A deep-learning web app that classifies movie reviews as **Positive** or **Negative**, built with TensorFlow/Keras and served with Streamlit. It also **visualizes which words drove the prediction**, so you can see *why* the model made its decision.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)

---

## ✨ Features

- **Real-time sentiment classification** — type a review, get Positive/Negative plus a confidence score.
- **Word-importance visualization** — each word is color-coded (🟩 green = pushed positive, 🟥 red = pushed negative) using occlusion-based attribution, so you can see the model's reasoning.
- **Confidence bar** for an at-a-glance read on how sure the model is.
- **Cached model loading** — the model loads once and is reused across interactions for a snappy UI.
- **Robust preprocessing** — out-of-vocabulary words are handled safely.

---

## 🧠 How the Model Works

The classifier processes a review through three stages:

```
Input: review text  →  Embedding  →  BiLSTM  →  Dense (sigmoid)  →  score (0–1)
       (word ids)      (vectors)    (memory)    (decision)
```

1. **Tokenize & encode** — the review is lowercased, split into words, and each word is mapped to an integer id from the IMDB vocabulary (top 10,000 words), then padded/truncated to 500 tokens.
2. **Embedding layer** — each word id becomes a learned 128-dimensional vector capturing its meaning.
3. **Bidirectional LSTM** — reads the sequence forward *and* backward, maintaining a memory of context so it understands phrases, negation, and long-range dependencies.
4. **Dense + sigmoid** — collapses the LSTM's summary into a single score between 0 and 1.
5. **Decision** — score **> 0.5 → Positive**, otherwise **Negative**.

### Why BiLSTM instead of SimpleRNN?
The original model used `SimpleRNN(activation='relu')`, which suffered from **exploding gradients** (training loss diverged into the billions). It was replaced with a **Bidirectional LSTM** (tanh activation) plus **gradient clipping** and **dropout**, which trains stably and generalizes better.

| | Original SimpleRNN | Current BiLSTM |
|---|---|---|
| Training stability | Diverged (loss → 3.6e10) | Stable convergence |
| Test accuracy | Not reliably measured | **~87%** |
| Architecture | Embedding → SimpleRNN(relu) → Dense | Embedding → BiLSTM(tanh) → Dropout → Dense |

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| [`main.py`](main.py) | The Streamlit web app (classification + word-importance visualization). |
| [`train.py`](train.py) | Clean, stable training script (BiLSTM + gradient clipping). Run this to retrain. |
| [`sentiment_model.h5`](sentiment_model.h5) | The trained model used by the app (~87% test accuracy). |
| [`simple_rnn_imdb.h5`](simple_rnn_imdb.h5) | Legacy SimpleRNN model (kept as a fallback / reference). |
| [`simplernn.ipynb`](simplernn.ipynb) | Original training notebook (shows the exploding-gradient problem). |
| [`prediction.ipynb`](prediction.ipynb) | Notebook for loading the model and testing predictions. |
| [`embedding.ipynb`](embedding.ipynb) | Standalone notebook explaining word embeddings. |
| [`requirements.txt`](requirements.txt) | Python dependencies for the app. |

> The app auto-loads `sentiment_model.h5` if present, and falls back to `simple_rnn_imdb.h5` otherwise.

---

## 🚀 Getting Started (Local)

### Prerequisites
- Python **3.11** (TensorFlow 2.15 supports Python 3.9–3.11)

### 1. Clone the repository
```bash
git clone https://github.com/abhishek-git94/IMDB-Movie-Review-Sentiment-Analysis.git
cd IMDB-Movie-Review-Sentiment-Analysis
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run main.py
```
The app opens at **http://localhost:8501**.

---

## 🔁 Retraining the Model

To retrain from scratch (downloads the IMDB dataset automatically):
```bash
python train.py
```
This produces a fresh `sentiment_model.h5`. Training uses:
- **Bidirectional LSTM** (64 units) with default tanh activation
- **Adam** optimizer with `clipnorm=1.0` (gradient clipping)
- **Dropout (0.5)** for regularization
- **EarlyStopping** on validation loss (restores best weights)

> Training on CPU takes ~20–30 minutes. A GPU is much faster but not required.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (already done).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. **Create app** → select this repository, branch `main`, main file `main.py`.
4. **Important:** In *Advanced settings*, set the **Python version to 3.11** (TensorFlow 2.15 does not support 3.12+).
5. Click **Deploy**. You'll get a public URL to share.

Every push to `main` automatically redeploys the app.

---

## 🛠️ Tech Stack

- **TensorFlow / Keras** — model definition, training, and inference
- **Streamlit** — interactive web UI
- **NumPy** — numerical processing
- **IMDB dataset** — 50,000 labeled movie reviews (25k train / 25k test)

---

## 📊 Dataset

The [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) contains 50,000 highly polar movie reviews, labeled positive or negative. Reviews come pre-encoded as integer sequences, where each integer represents a word ranked by frequency. This project uses the top **10,000** most frequent words.

---

## 📝 License

This project is for educational purposes. Feel free to use and adapt it.
