"""
Train a stable sentiment-analysis model on the IMDB dataset.

Fixes the issues in the original simplernn.ipynb:
  - Replaces SimpleRNN(activation='relu') -- prone to exploding gradients --
    with an LSTM (default tanh activation), which handles long sequences and
    keeps gradients stable.
  - Adds gradient clipping (clipnorm) to the optimizer as a second safeguard.
  - Saves in HDF5 (.h5) format, which loads reliably across environments
    (the .keras format can fail to load Bidirectional models elsewhere).

Run:  python train.py
Output:  sentiment_model.h5
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Hyperparameters (must match preprocessing in main.py)
MAX_FEATURES = 10000   # vocabulary size
MAX_LEN = 500          # sequence length
EMBED_DIM = 128
EPOCHS = 10
BATCH_SIZE = 64
OUTPUT_PATH = 'sentiment_model.h5'


def main():
    # 1. Load and pad the data
    print('Loading IMDB dataset...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    X_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')

    # 2. Build a stable model: Embedding -> BiLSTM -> Dense
    model = Sequential([
        Embedding(MAX_FEATURES, EMBED_DIM, input_length=MAX_LEN),
        Bidirectional(LSTM(64)),     # tanh by default -> no exploding gradients
        Dropout(0.5),                # regularization
        Dense(1, activation='sigmoid'),
    ])

    # clipnorm caps the gradient norm as an extra stability safeguard.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # 3. Train with early stopping on validation loss
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
    )

    # 4. Evaluate on the held-out test set
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\nTest loss: {loss:.4f}  |  Test accuracy: {acc:.4f}')

    # 5. Save in HDF5 format (robust to load across environments)
    model.save(OUTPUT_PATH)
    print(f'Saved model to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
