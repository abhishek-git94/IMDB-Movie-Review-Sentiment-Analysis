# Step 1: Import Libraries and Load the Model
import os
import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

MAX_FEATURES = 10000  # vocabulary size the model was trained on
MAX_LEN = 500         # sequence length the model expects

# Prefer the retrained BiLSTM model; fall back to the original SimpleRNN model.
# Use HDF5 (.h5): it loads more reliably across environments than the .keras
# format does for Bidirectional models.
MODEL_CANDIDATES = ['sentiment_model.h5', 'simple_rnn_imdb.h5']


# Step 2: Cached loaders (run once, reused across reruns)
@st.cache_resource
def load_word_index():
    """Load the IMDB word index and its reverse mapping (cached)."""
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index


@st.cache_resource
def load_sentiment_model():
    """Load the trained model once and keep it in memory (cached)."""
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return load_model(path), path
    raise FileNotFoundError(
        f"No model file found. Expected one of: {MODEL_CANDIDATES}"
    )


word_index, reverse_word_index = load_word_index()
model, model_path = load_sentiment_model()


# Step 3: Helper Functions
def decode_review(encoded_review):
    """Decode a list of integer ids back into readable text."""
    return ' '.join(reverse_word_index.get(i - 3, '?') for i in encoded_review)


def encode_words(words):
    """Map a list of words to clamped IMDB word ids."""
    encoded = []
    for word in words:
        # +3 matches Keras' reserved-index convention (0=pad, 1=start, 2=unknown).
        idx = word_index.get(word, 2) + 3
        # Clamp out-of-vocabulary words to the "unknown" token so the index
        # never exceeds the embedding's vocabulary size.
        if idx >= MAX_FEATURES:
            idx = 2
        encoded.append(idx)
    return encoded


def preprocess_text(text):
    """Turn raw user text into a padded, model-ready sequence."""
    encoded_review = encode_words(text.lower().split())
    return sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)


def predict_score(padded_batch):
    """Return the model's positive-sentiment score(s) for a padded batch."""
    return model.predict(padded_batch, verbose=0).reshape(-1)


def explain_prediction(words, baseline_score):
    """
    Occlusion-based word importance.

    For each word, hide it (replace with the 'unknown' token) and re-score the
    review. The change from the baseline tells us how much that word pushed the
    prediction:
        contribution = baseline_score - score_without_word
    Positive contribution -> hiding the word LOWERED the score, so the word was
    pushing the review POSITIVE.  Negative -> the word was pushing it NEGATIVE.
    """
    encoded = encode_words(words)
    if not encoded:
        return []

    # Build one occluded variant per word, then score them all in a single batch.
    variants = []
    for i in range(len(encoded)):
        masked = list(encoded)
        masked[i] = 2  # unknown token
        variants.append(masked)

    padded = sequence.pad_sequences(variants, maxlen=MAX_LEN)
    scores = predict_score(padded)
    contributions = baseline_score - scores
    return list(zip(words, contributions))


def render_importance_html(contributions):
    """Render words as colored chips: green = pushed positive, red = negative."""
    if not contributions:
        return ''
    max_abs = max(abs(c) for _, c in contributions) or 1e-9
    chips = []
    for word, c in contributions:
        alpha = min(1.0, abs(c) / max_abs)
        if c >= 0:  # pushed the score up -> positive
            color = f'rgba(0, 170, 0, {alpha:.3f})'
        else:       # pushed the score down -> negative
            color = f'rgba(220, 0, 0, {alpha:.3f})'
        chips.append(
            f'<span title="{c:+.4f}" style="background-color:{color};'
            f'padding:2px 5px;margin:2px;border-radius:4px;display:inline-block;">'
            f'{word}</span>'
        )
    return '<div style="line-height:2.2;">' + ' '.join(chips) + '</div>'


# Step 4: Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')
st.caption(f'Model in use: `{model_path}`')

user_input = st.text_area('Movie Review')
show_importance = st.checkbox('Show word importance', value=True)

if st.button('Classify'):
    if not user_input.strip():
        st.warning('Please enter a movie review first.')
    else:
        preprocessed_input = preprocess_text(user_input)
        score = float(predict_score(preprocessed_input)[0])
        sentiment = 'Positive' if score > 0.5 else 'Negative'

        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {score:.4f}')
        st.progress(score)

        if show_importance:
            st.subheader('Word importance')
            st.caption(
                'Each word is hidden in turn to measure its effect on the score. '
                'Greener = pushed the review more positive, redder = more negative '
                '(hover a word to see its exact contribution).'
            )
            contributions = explain_prediction(user_input.lower().split(), score)
            st.markdown(render_importance_html(contributions), unsafe_allow_html=True)
else:
    st.write('Please enter a movie review.')
