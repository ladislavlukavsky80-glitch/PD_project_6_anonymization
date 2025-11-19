# train_page_cnn.py
from pathlib import Path
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # pro vykreslení loss/accuracy křivek

IMG_SIZE = 256
IN_DIR = Path("preprocessed_pages")
LABELS_JSON = Path("config/page_labels_train.json")
MODEL_PATH = Path("models/page_cnn.h5")
HISTORY_JSON = Path("models/page_cnn_history.json")
HISTORY_PNG = Path("models/page_cnn_history.png")


def load_data():
    with open(LABELS_JSON, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)

    X = []
    y = []

    for fname, label in labels_dict.items():
        p = IN_DIR / fname
        if not p.exists():
            print(f"Varování: {p} neexistuje, přeskočeno.")
            continue

        img = cv2.imdecode(
            np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if img is None:
            print(f"Nelze načíst {p}, přeskočeno.")
            continue

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        X.append(img_resized)
        y.append(int(label))

    X = np.array(X, dtype=np.float32) / 255.0
    X = np.expand_dims(X, axis=-1)  # (N, H, W, 1)
    y = np.array(y, dtype=np.float32)

    print("Načteno:", X.shape[0], "obrázků")
    return X, y


def build_model():
    """
    CNN pro klasifikaci celé stránky:
    - 4 konvoluční bloky (Conv + MaxPool)
    - Dense 256 + Dropout
    - výstup: sigmoid (pravděpodobnost ANON)
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Conv blok 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Conv blok 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Conv blok 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Conv blok 4 (navíc – hlubší síť)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        # TADY byla chyba – používáme `name=` místo `class_name=`
        layers.Dense(256, activation='relu', name="dense_1"),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', name="output"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model



def plot_history(history):
    """Vykreslí a uloží graf z history (loss + accuracy)."""
    hist = history.history

    plt.figure(figsize=(10, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist:
        plt.plot(hof_hist := hist["val_loss"], label="val_loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy
    if "accuracy" in hist:
        plt.subplot(1, 2, 1 + 1)
        plt.plot(hist["accuracy"], label="train_acc")
        if "val_accuracy" in hist:
            plt.plot(hist["val_accuracy"], label="val_acc")
        plt.title("Training vs Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    HISTORY_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(HISTORY_PNG)
    print(f"Graf tréninku uložen do: {HISTORY_PNG}")
    plt.close()


def main():
    X, y = load_data()
    if X.shape[0] == 0:
        print("Žádná data pro trénink, zkontroluj LABELS_JSON a IN_DIR.")
        return

    print("Data:", X.shape, "pozitivních (ANON) =", int(y.sum()))

    # Train/val split (u malého datasetu spíš orientační)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )
    ]

    class_weight = {
        0: 18 / 7,  # cca 2.57
        1: 1.0
    }

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=80,  # dle požadavku vedoucího
        batch_size=8,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # Uložit model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model uložen do {MODEL_PATH}")

    # Uložit historii do JSON (aby se dala později analyzovat)
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history.history, f, ensure_ascii=False, indent=2)
    print(f"Historie tréninku uložena do {HISTORY_JSON}")

    # Vykreslit graf loss/accuracy
    plot_history(history)


if __name__ == "__main__":
    main()
