from pathlib import Path
import cv2
import numpy as np

# ---------- utils ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def imread_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path: Path, img_rgb: np.ndarray):
    """
    Očekává 3kanálový RGB (může být i "šedý" – R=G=B).
    """
    ensure_dir(path.parent)
    if img_rgb.ndim == 2:
        # pro jistotu: pokud by sem někdy doputoval 1 kanál, rozkopíruj ho
        img_rgb = np.stack([img_rgb, img_rgb, img_rgb], axis=-1)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


# ---------- fotometrie (z původního preprocesoru) ----------

def gray_world_white_balance(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb.astype(np.float32) + 1e-6
    mean = img.reshape(-1, 3).mean(axis=0)
    scale = mean.mean() / mean
    balanced = np.clip(img * scale, 0, 255).astype(np.uint8)
    return balanced

def clahe_luminance(img_rgb: np.ndarray, clip=2.0, tile=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def gamma_correct(img_rgb: np.ndarray, gamma=0.95) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return img_rgb
    inv = 1.0 / gamma
    lut = (np.linspace(0, 1, 256) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img_rgb, lut)

def denoise_bilateral(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img_rgb, d=5, sigmaColor=25, sigmaSpace=25)


# ---------- odhad náklonu + deskew ----------

def estimate_tilt_deg_vlines(img_rgb,
                             canny1=40, canny2=120,
                             hough_thr=80, min_len_ratio=0.18):
    """
    Hrubý odhad malého natočení (deg) podle svislých čar mřížky.
    Vrací *odchylku* od svislice (kladné = po směru hodinových ručiček).
    """
    import math
    h, w = img_rgb.shape[:2]
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    edges = cv2.Canny(g, canny1, canny2)

    min_len = int(min(h, w) * min_len_ratio)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 1800,
                            threshold=hough_thr,
                            minLineLength=min_len,
                            maxLineGap=6)
    if lines is None:
        return 0.0

    angs = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1 and abs(dy) < 1:
            continue
        ang = math.degrees(math.atan2(-dy, dx))  # 0° = vodorovně
        # téměř svislé: kolem 90° nebo -90°
        if abs(abs(ang) - 90.0) <= 8.0:
            dev = (90.0 - abs(ang))          # odchylka od 90°
            dev = dev if ang > 0 else -dev   # znaménko podle orientace
            angs.append(dev)

    if not angs:
        return 0.0

    return float(np.median(angs))


def deskew_for_anonymization(img_rgb: np.ndarray,
                             max_abs_rot_deg: float = 5.0,
                             min_effective_deg: float = 0.15):
    """
    Vrátí (out_img, tilt_deg, rotated_flag).

    - Nejprve odhadne náklon podle svislých linií.
    - Pokud je |tilt| < min_effective_deg → neotáčí (šum).
    - Pokud je |tilt| > max_abs_rot_deg → neotáčí (podezřelý dotazník).
    - Jinak otočí o tilt_deg kolem středu s bílým pozadím.
    """
    tilt_deg = estimate_tilt_deg_vlines(img_rgb)

    # příliš malá korekce = ignorujeme
    if abs(tilt_deg) < min_effective_deg:
        return img_rgb, tilt_deg, False

    # příliš velká korekce = necháme tak, jen zalogujeme
    if abs(tilt_deg) > max_abs_rot_deg:
        print(f"[WARN] {tilt_deg:.2f}° > {max_abs_rot_deg}° → NEOTÁČÍM (podezřelé zarovnání)")
        return img_rgb, tilt_deg, False

    h, w = img_rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), tilt_deg, 1.0)
    rot = cv2.warpAffine(
        img_rgb, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return rot, tilt_deg, True


# ---------- převod na černobílou + sjednocení tónu ----------

def normalize_to_gray(img_rgb: np.ndarray,
                      target_mean: float = 210.0,
                      target_std: float = 40.0) -> np.ndarray:
    """
    1) fotometrická normalizace (WB, CLAHE, gamma, denoise),
    2) převod na šedou,
    3) přeskalování na pevný mean/std → všechny stránky podobně "bílé".

    Výstup: 3kanálový RGB, ale R=G=B (vizuálně černobílé).
    """
    # 1) fotometrie v barvě (pomáhá odstranit barevný nádech papíru)
    x = gray_world_white_balance(img_rgb)
    x = clahe_luminance(x, clip=2.0, tile=(8, 8))
    x = gamma_correct(x, gamma=0.96)
    x = denoise_bilateral(x)

    # 2) převod na odstíny šedi
    g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 3) sjednocení tónu (normalizace mean/std)
    m = float(g.mean())
    s = float(g.std() + 1e-6)

    gn = (g - m) / s * target_std + target_mean
    gn = np.clip(gn, 0, 255).astype(np.uint8)

    # 3kanálová "šedá" – kompatibilní s imwrite_rgb a zbytkem pipeline
    g3 = np.stack([gn, gn, gn], axis=-1)
    return g3


# ---------- pipeline pro anonimizaci ----------

def preprocess_single_anon(page_path: Path,
                           max_abs_rot_deg: float = 5.0) -> np.ndarray:
    """
    Kompletní preproces pro anonimizaci:
    - načte stránku
    - změří náklon a případně otočí (pokud |tilt| ≤ max_abs_rot_deg)
    - převede na černobílý obrázek s jednotným tónem papíru
    """
    img = imread_rgb(page_path)

    img2, tilt_deg, rotated = deskew_for_anonymization(
        img,
        max_abs_rot_deg=max_abs_rot_deg
    )

    print(f"[PAGE] {page_path.name}: tilt={tilt_deg:.2f}°, rotated={rotated}")

    # černobílá + sjednocení světla/kontrastu
    img2 = normalize_to_gray(img2)

    return img2


def preprocess_all_anon(
    pages_dir="Original_pages",
    out_dir="preprocessed_anon",
    max_abs_rot_deg: float = 5.0,
    suffix=""
):
    """
    Batch pro anonimizaci: všechny PNG v pages_dir → zarovnané ČB PNG v out_dir.
    """
    pages_dir = Path(pages_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    page_files = sorted(p for p in pages_dir.glob("*.png"))
    if not page_files:
        raise RuntimeError(f"No PNG pages found in {pages_dir}")

    for p in page_files:
        try:
            out = preprocess_single_anon(
                p,
                max_abs_rot_deg=max_abs_rot_deg
            )
            out_name = p.stem + suffix + ".png"
            imwrite_rgb(out_dir / out_name, out)
            print(f"[OK] {p.name} -> {out_name}")
        except Exception as e:
            print(f"[FAIL] {p.name}: {e}")


if __name__ == "__main__":
    # typické spuštění:
    # python preprocess_anon.py
    preprocess_all_anon(
        pages_dir="Original_pages",
        out_dir="preprocessed_anon",
        max_abs_rot_deg=5.0
    )
