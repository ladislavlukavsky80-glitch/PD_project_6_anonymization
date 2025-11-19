# predict_pages_cnn.py
from pathlib import Path
import json
import numpy as np
import cv2
import tensorflow as tf

OUT_SCORES = Path("config/page_scores_cnn.json")

IMG_SIZE = 256
IN_DIR = Path("preprocessed_pages")
MODEL_PATH = Path("models/page_cnn.h5")
OUT_JSON = Path("config/page_labels_cnn.json")

# práh pro rozhodnutí – můžeš klidně později upravit (0.5 → default)
THRESH = 0.5


def preprocess_img(path: Path):
    """Načte obrázek, převede do grayscale, zmenší na 256×256 a normalizuje."""
    img = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE
    )
    if img is None:
        raise RuntimeError(f"Nelze načíst {path}")

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = img_resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, -1))  # tvar: (1, H, W, 1)
    return x


def main():
    if not IN_DIR.is_dir():
        print("Chyba: složka preprocessed_pages neexistuje.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model načten z {MODEL_PATH}")

    imgs = sorted(IN_DIR.glob("*.png"))
    if not imgs:
        print("Žádné PNG ve složce preprocessed_pages.")
        return

    labels = {}
    scores = {}

    for p in imgs:
        x = preprocess_img(p)
        prob = float(model.predict(x, verbose=0)[0, 0])
        do_anon = prob >= THRESH
        labels[p.name] = int(do_anon)
        scores[p.name] = prob

        print(f"{p.name:35}  prob_anon={prob:.4f}  →  {'ANON' if do_anon else 'COPY'}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    with open(OUT_SCORES, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print(f"\nHotovo – labely uloženy do {OUT_JSON}")
    print(f"Skóre (pravděpodobnosti) uloženy do {OUT_SCORES}")


if __name__ == "__main__":
    main()
