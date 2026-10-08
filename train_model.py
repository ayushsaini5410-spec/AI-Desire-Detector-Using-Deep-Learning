import csv, pickle, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "training_data.csv"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

texts, categories = [], []
with open(DATA_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        texts.append(row["text"].strip().lower())
        categories.append(row["category"].strip())

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories)
labels = list(label_encoder.classes_)

tokenizer = Tokenizer(num_words=3000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=30, padding="post", truncating="post")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Embedding(input_dim=min(3000, len(tokenizer.word_index) + 1), output_dim=32, input_length=30),
    GlobalAveragePooling1D(),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(len(labels), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
model.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=60, batch_size=8, callbacks=[early_stop], verbose=1
)
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
model.save(ROOT / "desire_detector.keras")
with open(ROOT / "tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open(ROOT / "labels.pkl", "wb") as f:
    pickle.dump(labels, f)
with open(ROOT / "training_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Validation accuracy: {accuracy:.4f}\\nValidation loss: {loss:.4f}\\n")
print(f"Training complete. Validation accuracy: {accuracy:.3f}")
print("Created desire_detector.keras, tokenizer.pkl, labels.pkl")
