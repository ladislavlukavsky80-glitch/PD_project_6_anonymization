import argparse
import json
import math
import os
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle


def load_existing_labels(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Povolíme jak bool, tak 0/1 apod.
        labels = {k: int(v) for k, v in data.items()}
        print(f"[manual_anon] Načteno {len(labels)} existujících labelů z {json_path}")
        return labels
    return {}


def save_labels(labels, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True) if os.path.dirname(json_path) else None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"[manual_anon] Uloženo {len(labels)} labelů do {json_path}")


def get_image_files(input_dir, ext_list=(".png", ".jpg", ".jpeg")):
    files = []
    for ext in ext_list:
        files.extend(glob(os.path.join(input_dir, f"*{ext}")))
    files = sorted(files)
    return files


def label_batch(image_paths, existing_labels):
    """
    Zobrazí jednu dávku obrázků v mřížce.
    Kliknutím na obrázek přepínáš stav anonymizace.
    Po zavření okna vrátí aktualizované labely.
    """
    n = len(image_paths)
    if n == 0:
        return existing_labels

    # Rozumné rozložení mřížky (skoro čtverec)
    cols = min(10, max(1, int(math.ceil(math.sqrt(n)))))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    fig.suptitle("Klikni na obrázek pro přepnutí ANON/NE-ANON. Zavři okno, až budeš hotový.", fontsize=14)

    # Když je jen jeden obrázek, axes není pole
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # Mapování axis -> index obrázku
    ax_to_idx = {}
    rects = {}

    for idx, img_path in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        ax_to_idx[ax] = idx

        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis("off")

        fname = os.path.basename(img_path)
        current_label = existing_labels.get(fname, 0)  # 0 = default NE-ANON

        # Titulek s informací
        title = f"{idx}: {fname}"
        if current_label == 1:
            title += " [ANON]"
        ax.set_title(title, fontsize=8)

        # Rámeček pro zvýraznění
        h, w = img.shape[0], img.shape[1]
        rect = Rectangle(
            (0, 0),
            w,
            h,
            linewidth=2,
            edgecolor="red" if current_label == 1 else "black",
            facecolor="none",
        )
        ax.add_patch(rect)
        rects[ax] = rect

    # Skryj prázdné osy
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    # Stav vybraných
    selected = set(
        idx for idx, img_path in enumerate(image_paths)
        if existing_labels.get(os.path.basename(img_path), 0) == 1
    )

    def on_click(event):
        ax = event.inaxes
        if ax not in ax_to_idx:
            return
        idx = ax_to_idx[ax]
        fname = os.path.basename(image_paths[idx])

        if idx in selected:
            selected.remove(idx)
            existing_labels[fname] = 0
        else:
            selected.add(idx)
            existing_labels[fname] = 1

        # Aktualizace titulku
        base_title = f"{idx}: {fname}"
        if existing_labels[fname] == 1:
            base_title += " [ANON]"
        ax.set_title(base_title, fontsize=8)

        # Aktualizace rámečku
        rect = rects[ax]
        rect.set_edgecolor("red" if existing_labels[fname] == 1 else "black")
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return existing_labels


def main():
    parser = argparse.ArgumentParser(
        description="Ruční označení stránek k anonimizaci (vytvoří page_labels_cnn.json)."
    )
    parser.add_argument(
        "--in", dest="input_dir", required=True,
        help="Adresář s předzpracovanými stránkami (např. preprocessed_anon)."
    )
    parser.add_argument(
        "--out", dest="json_out", default="page_labels_cnn.json",
        help="Cesta k výstupnímu JSON souboru (default: page_labels_cnn.json)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Počet stránek v jedné dávce (default: 50)."
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    json_out = args.json_out
    batch_size = args.batch_size

    if not os.path.isdir(input_dir):
        print(f"[manual_anon] Chyba: adresář {input_dir} neexistuje.")
        return

    all_files = get_image_files(input_dir)
    if not all_files:
        print(f"[manual_anon] V adresáři {input_dir} nebyly nalezeny žádné obrázky.")
        return

    print(f"[manual_anon] Nalezeno {len(all_files)} souborů v {input_dir}")

    labels = load_existing_labels(json_out)

    # Vytvoř seznam souborů, které ještě nemají label
    print(f"[manual_anon] Nalezeno {len(all_files)} souborů v {input_dir}")

    labels = load_existing_labels(json_out)

    # NOVĚ: zobrazujeme všechny soubory, i ty už jednou označené
    files_to_show = all_files
    print(f"[manual_anon] Celkem souborů k zobrazení: {len(files_to_show)}")

    i = 0
    while i < len(files_to_show):
        batch = files_to_show[i:i + batch_size]
        print(f"[manual_anon] Dávka {i + 1}–{i + len(batch)} z {len(files_to_show)}")

        labels = label_batch(batch, labels)
        save_labels(labels, json_out)

        i += batch_size
        if i < len(files_to_show):
            ans = input("[manual_anon] Pokračovat na další dávku? [Enter = ano, q = konec] ")
            if ans.strip().lower() == "q":
                break

    print("[manual_anon] Hotovo.")


if __name__ == "__main__":
    main()
