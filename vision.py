# vision.py
import re
import easyocr
import io, os, time
import cv2
import numpy as np
from PIL import Image

DEFAULT_SCALES = np.linspace(0.7, 1.3, 21)  # multi-scale biar tahan beda DPI
DEFAULT_THRESHOLD = 0.85

_reader = easyocr.Reader(['en'])

class ImageTapper:
    """
    Util untuk:
      - Screenshot dari Appium driver
      - Template matching multi-scale (OpenCV)
      - Tap ke tengah hasil match
    """
    def __init__(self, driver, template_dir="./templates",
                 threshold: float = DEFAULT_THRESHOLD,
                 scales = DEFAULT_SCALES):
        self.driver = driver
        self.template_dir = template_dir
        self.threshold = threshold
        self.scales = scales

    # ---------- Low-level ----------

    def _screenshot_bgr(self) -> np.ndarray:
        """Ambil screenshot dari driver → ndarray BGR (OpenCV)."""
        png = self.driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png)).convert("RGB")
        return np.array(img)[:, :, ::-1]  # RGB → BGR

    def _read_template(self, name_or_path: str) -> np.ndarray | None:
        """Baca template dari path absolut/relatif atau key (tanpa .png)."""
        path = name_or_path
        if not os.path.exists(path):
            # coba "<template_dir>/<name>.png"
            path = os.path.join(self.template_dir, f"{name_or_path}.png")
        if not os.path.exists(path):
            return None
        templ = cv2.imread(path, cv2.IMREAD_COLOR)
        return templ

    def _multiscale_match(self, haystack_bgr: np.ndarray, needle_bgr: np.ndarray):
        """Return (best_score, top_left_xy, bottom_right_xy, used_scale)."""
        best = (-1.0, None, None, None)
        for s in self.scales:
            tw = max(5, int(needle_bgr.shape[1] * s))
            th = max(5, int(needle_bgr.shape[0] * s))
            templ = cv2.resize(needle_bgr, (tw, th),
                               interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)

            res = cv2.matchTemplate(haystack_bgr, templ, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best[0]:
                tl = max_loc
                br = (tl[0] + templ.shape[1], tl[1] + templ.shape[0])
                best = (float(max_val), tl, br, float(s))
        return best

    # ---------- Public API ----------

    def find(self, template_name: str):
        """
        Cari template pada layar.
        Return dict: {ok, score, center(x,y), rect((x1,y1),(x2,y2)), scale}
        """
        templ = self._read_template(template_name)
        if templ is None:
            return {"ok": False, "reason": f"template not found: {template_name}"}

        screen = self._screenshot_bgr()
        score, tl, br, scale = self._multiscale_match(screen, templ)
        if tl is None:
            return {"ok": False, "score": score, "reason": "no location"}

        cx = (tl[0] + br[0]) // 2
        cy = (tl[1] + br[1]) // 2
        return {
            "ok": score >= self.threshold,
            "score": score,
            "center": (cx, cy),
            "rect": (tl, br),
            "scale": scale,
        }

    def tap_center(self, center_xy: tuple[int, int]):
        """Tap menggunakan Appium mobile: clickGesture."""
        x, y = center_xy
        self.driver.execute_script("mobile: clickGesture", {"x": int(x), "y": int(y)})

    def tap_image(self, template_name: str, retries: int = 3, delay: float = 0.6):
        """
        Cari lalu tap gambar. Balikkan (ok, score, center).
        Auto-retry ketika score < threshold.
        """
        last = None
        for _ in range(retries):
            res = self.find(template_name)
            last = res
            if res.get("ok") and res.get("center"):
                self.tap_center(res["center"])
                return True, res.get("score", 0.0), res["center"]
            time.sleep(delay)
        return False, (last or {}).get("score", 0.0), (last or {}).get("center")
    
    # ---------- OCR: baca ID di ROI persentase ----------
    def read_user_id(self, roi_pct=(0.36, 0.33, 0.95, 0.62)):
        """
        Baca ID dari area tertentu (persentase layar).
        roi_pct: (x1p, y1p, x2p, y2p) dalam 0..1
        return: string angka ID atau None
        """
        if _reader is None:
            return None

        screen = self._screenshot_bgr()
        h, w = screen.shape[:2]
        x1 = int(w * roi_pct[0]); y1 = int(h * roi_pct[1])
        x2 = int(w * roi_pct[2]); y2 = int(h * roi_pct[3])
        roi = screen[y1:y2, x1:x2]

        # preprocessing ringan untuk stabilisasi
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # OCR
        results = _reader.readtext(th)
        text_all = " ".join([t for _, t, _ in results]) if results else ""
        # cari pola "ID:" atau angka panjang
        m = re.search(r'\bID[:\s]*([0-9]{5,})\b', text_all, flags=re.I)
        if m:
            return m.group(1)
        # fallback: ambil angka terpanjang
        nums = re.findall(r'[0-9]{6,}', text_all)
        return max(nums, key=len) if nums else None

    def read_user_id_near_tukar(self, tukar_template="tombol_tukar"):
        """
        Temukan tombol TUKAR via self.find(), lalu OCR area di kiri tombol pada baris yang sama.
        tukar_template: boleh nama template tanpa .png (disimpan di template_dir) atau path lengkap.
        Return: string angka ID atau None.
        """
        if _reader is None:
            return None

        img = self._screenshot_bgr()

        # Pakai util yang sudah ada:
        res = self.find(tukar_template)  # akan baca template dari template_dir bila nama saja
        rect = res.get("rect")           # rect = ((x1,y1),(x2,y2))
        if not rect:
            # fallback ke ROI persentase kalau tombol tak ketemu sama sekali
            return self.read_user_id()

        (x1, y1), (x2, y2) = rect  # posisi tombol TUKAR
        h, w = img.shape[:2]

        # Ambil strip horizontal setinggi tombol, agak ditambah padding atas-bawah
        pad_y_top = int(0.15 * (y2 - y1))
        pad_y_bot = int(0.20 * (y2 - y1))
        band_y1 = max(0, y1 - pad_y_top)
        band_y2 = min(h, y2 + pad_y_bot)

        # Melebar cukup jauh ke kiri untuk menangkap "ID: 2194..." dan nama
        left_margin = max(0, x1 - int(0.48 * w))  # ~ setengah layar ke kiri
        roi = img[band_y1:band_y2, left_margin:x1]

        # ----- preprocessing OCR -----
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        scale = 1.6
        up = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)))
        th = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)

        results = _reader.readtext(th)
        text_all = " ".join([t for _, t, _ in results]) if results else ""

        m = re.search(r'\bID[:\s]*([0-9]{5,})\b', text_all, flags=re.I)
        if m:
            return m.group(1)

        nums = re.findall(r'[0-9]{6,}', text_all)
        return max(nums, key=len) if nums else None

    def read_user_id_pixels(self, roi_px, save_debug=False):
        """
        OCR 'ID: xxxxx' dari ROI pixel absolut.
        roi_px: (x1, y1, x2, y2) pada full screenshot.
        Selalu simpan debug ROI kalau save_debug=True, bahkan jika EasyOCR belum siap.
        """
        import os, cv2, re, numpy as np

        # --- Lokasi folder debug (pasti ada) ---
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DEBUG_DIR = os.path.join(BASE_DIR, "debug")
        os.makedirs(DEBUG_DIR, exist_ok=True)

        # --- Ambil screenshot & crop ROI ---
        img = self._screenshot_bgr()
        x1, y1, x2, y2 = map(int, roi_px)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
        roi = img[y1:y2, x1:x2]

        # --- Simpan RAW ROI dulu untuk verifikasi area ---
        if save_debug:
            raw_path = os.path.join(DEBUG_DIR, "debug_roi_pixels_raw.png")
            cv2.imwrite(raw_path, roi)
            print("[DEBUG] ROI raw saved to:", raw_path, flush=True)

        # --- Preprocess untuk OCR ---
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        scale = 1.8
        up = cv2.resize(gray, (int(gray.shape[1]*scale), int(gray.shape[0]*scale)))
        th = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)

        if save_debug:
            proc_path = os.path.join(DEBUG_DIR, "debug_roi_pixels_proc.png")
            cv2.imwrite(proc_path, th)
            print("[DEBUG] ROI proc saved to:", proc_path, flush=True)
            print("[DEBUG] CWD is:", os.getcwd(), flush=True)

        # --- Kalau OCR engine belum siap, keluar sekarang ---
        if _reader is None:
            print("[DEBUG] EasyOCR not initialized (_reader is None). Install 'easyocr'?", flush=True)
            return None

        # --- OCR ---
        results = _reader.readtext(th)
        text_all = " ".join([t for _, t, _ in results]) if results else ""
        m = re.search(r'\bID[:\s]*([0-9]{5,})\b', text_all, flags=re.I)
        if m:
            return m.group(1)

        nums = re.findall(r'[0-9]{6,}', text_all)
        return max(nums, key=len) if nums else None

    
    def scale_roi_from_base(self, base_w, base_h, roi_px_base):
        """
        Mengubah ROI pixel dari resolusi dasar (base_w x base_h) ke resolusi
        screenshot saat ini.
        """
        img = self._screenshot_bgr()
        cur_h, cur_w = img.shape[:2]

        x1, y1, x2, y2 = roi_px_base
        sx = cur_w / float(base_w)
        sy = cur_h / float(base_h)

        return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
