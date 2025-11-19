from pathlib import Path
import argparse
import json
import cv2
import numpy as np

# ============================
# I/O pomocné funkce
# ============================

def imread_gray(path: Path):
    """Načte obrázek jako grayscale (bez problémů s UTF-8 cestou)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Nelze načíst obrázek: {path}")
    return img


def imread_rgb(path: Path):
    """Načte obrázek jako RGB."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Nelze načíst obrázek: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path: Path, img_bgr_or_gray):
    """Bezpečný zápis obrázku na Windows (UTF-8 cesty). Výsledek je vždy PNG."""
    ext = path.suffix.lower()
    ok, buf = cv2.imencode(ext, img_bgr_or_gray)
    if not ok:
        raise RuntimeError(f"Encode selhal: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


# ============================
# Geometrie ROI
# ============================

def scale_rois(rois, src_wh, templ_wh):
    """
    Přepočet ROI z rozměru šablony (tw,th) na aktuální obrázek (sw,sh).
    rois: list(dict(x,y,w,h)) v souřadnicích šablony.
    """
    tw, th = templ_wh
    sw, sh = src_wh
    sx = sw / tw
    sy = sh / th
    out = []
    for r in rois:
        out.append({
            "x": int(round(r["x"] * sx)),
            "y": int(round(r["y"] * sy)),
            "w": int(round(r["w"] * sx)),
            "h": int(round(r["h"] * sy)),
        })
    return out


def apply_blackout_rois(img_bgr, rois, color=(255, 255, 255)):
    """
    Zakryje zadané ROI plnou barvou (white/black/gray).
    img_bgr: BGR obraz (OpenCV formát).
    rois: list(dict(x,y,w,h)) v souřadnicích aktuálního obrázku.
    """
    out = img_bgr.copy()
    for r in rois:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness=-1)
    return out


# ============================
# Načtení JSONů
# ============================

def load_anon_rois(json_path: Path):
    """
    Načte anonymizační ROI ve formátu gui_define_anon_roi:
    {
      "template_image": ".",
      "rois": [
        {"name": ".", "type": "blackout", "bbox": [x,y,w,h]},
        .
      ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    rois = []
    for r in cfg.get("rois", []):
        x, y, w, h = r["bbox"]
        rois.append({"x": x, "y": y, "w": w, "h": h})
    return rois


def load_page_labels(json_path: Path):
    """
    Načte výstupy CNN (rozhodnutí o anonymizaci):
    {
      "IPDdiaries_scan_01.png": 0,
      "IPDdiaries_scan_07.png": 1,
      ...
    }
    kde 1 = ANON (na stránce je jméno / identifikace, anonymizujeme),
        0 = COPY (stránka bez osobních údajů).
    """
    if not json_path.exists():
        print(f"Varování: {json_path} neexistuje, všechny stránky budou COPY.")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels


# ============================
# Pomocné formátování jmen
# ============================

def format_patient_id(idx: int) -> str:
    """P0001, P0002, ..."""
    return f"P{idx:04d}"


def format_page_id(idx: int) -> str:
    """N01, N02, ..."""
    return f"N{idx:02d}"


def get_label_for_filename(fname: str, labels: dict) -> int:
    """
    Vrátí label (0/1) pro daný vstupní soubor.

    1) zkusí přesné jméno (např. IPDdiaries_scan_01.png)
    2) fallbacky pro staré `_anon` apod. necháme klidně, ale
       protože už `_anon` nebudeme zpracovávat, mělo by to být zbytečné.
    """
    if fname in labels:
        return labels[fname]

    # historický fallback – kdyby se tam přece jen objevil nějaký "_anon"
    if fname.endswith("_anon.png"):
        base = fname.replace("_anon", "")
        if base in labels:
            return labels[base]

    return 0


# ============================
# Hlavní pipeline
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="preprocessed_anon", type=str,
                    help="Složka se znormalizovanými stránkami")
    ap.add_argument("--templates", default="Templates", type=str,
                    help="Složka se šablonami")
    ap.add_argument("--template07", default="IPDdiaries_scan_07_norm.png", type=str,
                    help="Šablona 07 (podle ní jsou definovány anonymizační ROI)")
    ap.add_argument("--anon-json", default="config/anon_header_rois.json", type=str,
                    help="JSON s ROI hlavičky (souřadnice v šabloně 07)")
    ap.add_argument("--labels-json", default="config/page_labels_cnn.json", type=str,
                    help="JSON s rozhodnutími CNN (1=ANON, 0=COPY)")
    ap.add_argument("--out-dir", default="results", type=str,
                    help="Cílová složka pro výstupy")
    ap.add_argument("--mask-color", default="white", choices=["black", "white", "gray"],
                    help="Barva masky pro blackout")
    ap.add_argument("--save-debug", action="store_true",
                    help="Uloží vedle výstupu i TXT s akcí (ANON/COPY) a novým jménem")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    t_dir = Path(args.templates)

    print(f"[INFO] Vstupní složka: {in_dir.resolve()}")
    print(f"[INFO] Výstupní složka: {out_dir.resolve()}")

    # Šablona pro rozměry (normovaná 07)
    t07_path = t_dir / args.template07
    print(f"[INFO] Načítám šablonu: {t07_path}")
    t07_gray = imread_gray(t07_path)
    th, tw = t07_gray.shape  # výška/šířka šablony

    # Anonymizační ROI v souřadnicích šablony
    anon_rois_template = load_anon_rois(Path(args.anon_json))

    # CNN labely (0/1)
    page_labels = load_page_labels(Path(args.labels_json))

    # Barva masky
    if args.mask_color == "white":
        mask_color = (255, 255, 255)
    elif args.mask_color == "gray":
        mask_color = (200, 200, 200)
    else:
        mask_color = (0, 0, 0)

    # *** KLÍČOVÁ ÚPRAVA ***
    # Bereme jen soubory bez "_anon" ve jménu (abychom neměli duplicitní stránky).
    all_pngs = sorted(in_dir.glob("*.png"))
    imgs = [p for p in all_pngs if "_anon" not in p.stem]

    if not imgs:
        raise RuntimeError(f"Ve složce {in_dir} nejsou žádné vhodné *.png stránky.")

    print(f"[INFO] Nalezeno {len(imgs)} PNG souborů pro zpracování v {in_dir}")

    # čítače Pxxxx / Nyy
    patient_idx = 0
    page_idx = 0

    for p in imgs:
        rgb = imread_rgb(p)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # rozhodnutí o anonymizaci podle CNN labelu
        label_val = get_label_for_filename(p.name, page_labels)
        do_anon = bool(label_val)
        action = "ANON" if do_anon else "COPY"

        # logika jmen Pxxxx_Nyy
        if do_anon:
            # nová hlavičková stránka → nový pacient, stránka N01
            patient_idx += 1
            page_idx = 1
        else:
            # běžná stránka:
            #  - pokud ještě žádný pacient není, začni P0001_N01
            #  - jinak zvedej jen N
            if patient_idx == 0:
                patient_idx = 1
                page_idx = 1
            else:
                page_idx += 1

        patient_id = format_patient_id(patient_idx)
        page_id = format_page_id(page_idx)
        new_name = f"{patient_id}_{page_id}.png"
        out_path = out_dir / new_name

        if do_anon:
            H, W = gray.shape
            rois = scale_rois(anon_rois_template, (W, H), (tw, th))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            anonym = apply_blackout_rois(bgr, rois, color=mask_color)
            imwrite(out_path, anonym)
        else:
            # jen překopírujeme s novým názvem
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            imwrite(out_path, bgr)

        print(f"[{action}] {p.name} -> {new_name} (P={patient_id}, page={page_id})")

        if args.save_debug:
            dbg = {
                "action": action,
                "orig_name": p.name,
                "new_name": new_name,
                "patient_id": patient_id,
                "page_id": page_id,
                "do_anon": do_anon,
                "raw_label": int(label_val),
            }
            (out_dir / f"{p.stem}_anon.txt").write_text(
                json.dumps(dbg, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )


if __name__ == "__main__":
    main()
