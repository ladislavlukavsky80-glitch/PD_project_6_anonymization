from pathlib import Path
import json
import cv2
import numpy as np

TEMPLATE_IMG = Path("Templates/IPDdiaries_scan_07.png")  # šablona 1. strany
OUT_JSON = Path("config/anon_header_rois.json")
PREVIEW_SRC = Path("preprocessed_pages")  # náhled anonymizace se dělá na 1. stránce z této složky

WIN_MAIN = "Anon ROI (táhni myší; ENTER=potvrdit, SPACE=zrušit, U=undo, A=preview on/off, ESC=uložit a konec)"
WIN_PREV = "Preview (A=toggle) — anonymizace zón"

# ===== Pomocné =====

def _safe_destroy(win_name: str):
    try:
        # vrací -1, pokud okno neexistuje; >=0 pokud existuje
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(win_name)
    except cv2.error:
        pass


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def imread_rgb(p: Path):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Nelze načíst: {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imwrite_bgr(p: Path, img_bgr):
    ok, buf = cv2.imencode(p.suffix, img_bgr)
    if not ok:
        raise RuntimeError(f"Encode selhal: {p}")
    ensure_dir(p)
    buf.tofile(str(p))

def rect_overlap(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    return inter_w > 0 and inter_h > 0

def any_overlap(bbox, boxes):
    return any(rect_overlap(bbox, q) for q in boxes)

def blackout(img_rgb, boxes, color=(0,0,0)):
    out = img_rgb.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(out, (x,y), (x+w, y+h), color, thickness=-1)
    return out

# ===== Interaktivní výběr =====

class RoiMaker:
    def __init__(self, base_rgb, scale_max_w=1200, scale_max_h=900):
        self.base = base_rgb
        H, W = base_rgb.shape[:2]
        self.scale = min(scale_max_w / W, scale_max_h / H, 1.0)
        self.disp = cv2.resize(base_rgb, (int(W*self.scale), int(H*self.scale)))
        self.clone = self.disp.copy()
        self.boxes = []   # [(x,y,w,h)] v PŮVODNÍCH pixelech šablony
        self.drawing = False
        self.pt0 = None
        self.rect_disp = None
        self.preview_on = False
        self.zoom_step = 0.1

        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN_MAIN, self._mouse)

        # připrav náhledový obrázek (první stránka z preprocessed_pages)
        self.prev_img_rgb = self._find_preview_image()

    def _find_preview_image(self):
        if PREVIEW_SRC.is_dir():
            pages = sorted(PREVIEW_SRC.glob("*.png"))
            if pages:
                try:
                    return imread_rgb(pages[0])
                except Exception:
                    pass
        return None  # dříve: self.base.copy()

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt0 = (x, y)
            self.rect_disp = None
            self.disp = self.clone.copy()
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.disp = self.clone.copy()
            x0, y0 = self.pt0
            cv2.rectangle(self.disp, (x0,y0), (x,y), (0,255,0), 2)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x0, y0 = self.pt0; x1, y1 = x, y
            x_min, y_min = min(x0,x1), min(y0,y1)
            w = abs(x1 - x0); h = abs(y1 - y0)
            if w > 0 and h > 0:
                self.rect_disp = (x_min, y_min, w, h)
                self.disp = self.clone.copy()
                cv2.rectangle(self.disp, (x_min,y_min), (x_min+w,y_min+h), (0,255,0), 2)

    def _disp_to_src(self, rect):
        x, y, w, h = rect
        inv = 1.0 / self.scale
        return (int(round(x*inv)), int(round(y*inv)),
                int(round(w*inv)), int(round(h*inv)))

    def _redraw_overlay(self):
        self.disp = self.clone.copy()
        # vykresli uložené boxy v displejových souřadnicích
        for (sx,sy,sw,sh) in self.boxes:
            x = int(round(sx * self.scale))
            y = int(round(sy * self.scale))
            w = int(round(sw * self.scale))
            h = int(round(sh * self.scale))
            cv2.rectangle(self.disp, (x,y), (x+w,y+h), (0,140,255), 2)

    def _update_preview(self):
        if not self.preview_on or self.prev_img_rgb is None:
            _safe_destroy(WIN_PREV)  # dříve: cv2.destroyWindow(WIN_PREV)
            return
        # 1) připrav maskovaný náhled v rozměru šablony (boxy jsou v souřadnicích šablony)
        Hs, Ws = self.base.shape[:2]
        img_src = cv2.resize(self.prev_img_rgb, (Ws, Hs), interpolation=cv2.INTER_AREA)
        masked_src = blackout(img_src, self.boxes, color=(0, 0, 0))
        # 2) zmenši na stejnou velikost jako editor
        dh, dw = self.disp.shape[:2]
        masked_disp = cv2.resize(masked_src, (dw, dh), interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN_PREV, cv2.cvtColor(masked_disp, cv2.COLOR_RGB2BGR))

    def loop(self):
        while True:
            self._update_preview()
            cv2.imshow(WIN_MAIN, cv2.cvtColor(self.disp, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(20) & 0xFF

            # zoom
            if key in (43, ord('=')):  # '+'
                self.scale = min(1.0, self.scale * (1 + self.zoom_step))
                self.disp = cv2.resize(self.base, (int(self.base.shape[1]*self.scale),
                                                   int(self.base.shape[0]*self.scale)))
                self.clone = self.disp.copy()
                self._redraw_overlay()
                continue
            if key == 45:  # '-'
                self.scale = max(0.3, self.scale * (1 - self.zoom_step))
                self.disp = cv2.resize(self.base, (int(self.base.shape[1]*self.scale),
                                                   int(self.base.shape[0]*self.scale)))
                self.clone = self.disp.copy()
                self._redraw_overlay()
                continue

            if key == ord('a'):  # toggle preview
                self.preview_on = not self.preview_on
                if not self.preview_on:
                    _safe_destroy(WIN_PREV)  # dříve: cv2.destroyWindow(WIN_PREV)
                continue

            if key == ord('u'):  # undo
                if self.boxes:
                    self.boxes.pop()
                    self._redraw_overlay()
                continue

            if key == 32:  # SPACE zruš aktuální kreslení
                self.rect_disp = None
                self.disp = self.clone.copy()
                self._redraw_overlay()
                continue

            if key == 13:  # ENTER potvrdit výběr
                if self.rect_disp is not None:
                    cand = self._disp_to_src(self.rect_disp)
                    # zákaz překryvu
                    if any_overlap(cand, self.boxes):
                        # vizuální upozornění (blik červený)
                        tmp = self.disp.copy()
                        x,y,w,h = self.rect_disp
                        cv2.rectangle(tmp, (x,y), (x+w,y+h), (0,0,255), 3)
                        cv2.imshow(WIN_MAIN, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(300)
                        # neuložíme
                    else:
                        self.boxes.append(cand)
                        self._redraw_overlay()
                    self.rect_disp = None
                continue

            if key == 27:  # ESC = uložit a konec
                break

        cv2.destroyAllWindows()
        return self.boxes

# ===== Uložení/čtení JSON =====

def save_json(template_img: Path, boxes, out_json: Path):
    cfg = {
        "template_image": str(template_img),
        "rois": [{"name": f"anon_{i+1}", "type": "blackout", "bbox": [x,y,w,h]}
                 for i, (x,y,w,h) in enumerate(boxes)]
    }
    ensure_dir(out_json)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"✅ Uloženo: {out_json}  (zón: {len(boxes)})")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=str, default=str(TEMPLATE_IMG),
                    help="Cesta k šabloně pro definici ROI")
    ap.add_argument("--max-width", type=int, default=900, help="Max šířka okna")
    ap.add_argument("--max-height", type=int, default=700, help="Max výška okna")
    args = ap.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"❌ Nenalezena šablona: {template_path}")
        return

    base = imread_rgb(template_path)
    maker = RoiMaker(base, scale_max_w=args.max_width, scale_max_h=args.max_height)
    boxes = maker.loop()
    save_json(template_path, boxes, OUT_JSON)

if __name__ == "__main__":
    main()
