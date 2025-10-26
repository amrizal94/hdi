automation_running = False
user_id_global = ""
produk_global = ""

import unittest
import requests
from appium import webdriver
from appium.options.android import UiAutomator2Options
# from appium.webdriver.common.appiumby import AppiumBy
import os
import time

import cv2, io
import numpy as np
from PIL import Image
from vision import ImageTapper  # >>>

TEMPLATE_DIR = "./templates"
PRODUCT_TEMPLATES = {
    "zeus": os.path.join(TEMPLATE_DIR, "zeus.png"),   # siapkan crop kecil yang unik
    "kucing": os.path.join(TEMPLATE_DIR, "kucing.png"),
    "mahjong": os.path.join(TEMPLATE_DIR, "mahjong.png"),
}
MATCH_THRESHOLD = 0.82
SCALES = (0.7, 0.8, 0.9, 1.0, 1.1)  # multi-scale

capabilities = dict(
    platformName='Android',
    automationName='uiautomator2',
    deviceName='Android',
    appPackage='com.neptune.domino',
    appActivity='com.pokercity.lobby.lobby',
    language='en',
    locale='US',
    noReset=True,
)

appium_server_url = 'http://localhost:4723'

# ==== Referensi koordinat & resolusi (patokan) ====
# Dari diskusi:
#   - X_ref = 1088 saat W_ref = 1600
#   - Y_ref = 65  saat H_ref = 900
REF_W, REF_H = 1600, 900 # resolusi patokan ketika develop
targets = {
    # "tukar_icon": (1088, 65),
    "tukar_icon": (977, 65),
    "input_id_teman": (940, 220),
    "tombol_cari": (1300, 220),
    "tombol_tukar": (1290, 350),
    "input_sandi": (900, 430),
    "tombol_tentukan": (800, 630),
    "mahjong": (1400, 300),
    "kucing": (1400, 500),
    "zeus": (1400, 680),
    "tukar_kartu": (225, 425),
    "tentukan_kartu": (800, 785),
}

KEYCODE_MAP = {
    **{c: k for c, k in zip("abcdefghijklmnopqrstuvwxyz", range(29, 55))},
    **{c: k for c, k in zip("0123456789", range(7, 17))},
    '@': 77, '#': 18, '*': 17, '+': 81, ',': 55, '-': 69, '.': 56, '/': 76,
    '=': 70, '_': 69, ' ': 62, '!': 8, '"': 12, '$': 11, '%': 12, '&': 14,
    '(': 15, ')': 16, '?': 76, ':': 74, ';': 74, '\'': 75, '"': 75,
}

# ==== Opsi pembulatan koordinat ====
# "round" | "floor" | "ceil"
ROUNDING_MODE = os.getenv("ROUNDING_MODE", "round")

def scale_value(ref_val: float, ref_range: float, now_range: float) -> float:
    """Skalakan satu dimensi (X atau Y)."""
    return (ref_val / ref_range) * now_range

def round_coord(val: float) -> int:
    """Pembulatan koordinat sesuai mode."""
    if ROUNDING_MODE == "floor":
        import math
        return math.floor(val)
    if ROUNDING_MODE == "ceil":
        import math
        return math.ceil(val)
    # default: round ke integer terdekat
    return round(val)

def is_appium_running() -> bool:
    for path in ("/status", "/wd/hub/status"):
        try:
            r = requests.get(f"{appium_server_url}{path}", timeout=2)
            if r.status_code == 200:
                # Appium 2 biasanya ada kunci 'value' di json
                # Appium 1 ada 'status' legacy; dua-duanya kita terima.
                try:
                    j = r.json()
                    if isinstance(j, dict) and ("value" in j or "status" in j):
                        return True
                except Exception:
                    # kalau bukan JSON tapi 200, tetap anggap OK
                    return True
        except Exception:
            pass
    return False

class TestAppium(unittest.TestCase):
    resolutionX = 0
    resolutionY = 0
    img: ImageTapper | None = None  # >>>
    
    def setUp(self) -> None:
        self.driver = webdriver.Remote(appium_server_url, options=UiAutomator2Options().load_capabilities(capabilities))
        self.resolutionX = self.driver.get_window_size()['width']
        self.resolutionY = self.driver.get_window_size()['height']
        # ⬇️ INISIALISASI ImageTapper
        self.img = ImageTapper(self.driver, template_dir=TEMPLATE_DIR, threshold=MATCH_THRESHOLD)

    def tearDown(self) -> None:
        if self.driver:
            self.driver.quit()

    def type_text_via_keycode(self, text: str):
        for char in text:
            key = KEYCODE_MAP.get(char.lower())
            if key:
                meta = 1 if char.isupper() else 0
                self.driver.press_keycode(key, metastate=meta)
            else:
                print(f"Character '{char}' not mapped")
                return
            time.sleep(0.1)  # jeda antar karakter
    
    def _compute_scaled_target(self, target_name: str = "tukar_icon") -> tuple[int, int, float, float]:
        """Hitung target (x,y) berdasarkan viewport saat ini."""
        x_f = scale_value(targets[target_name][0], REF_W, self.resolutionX)
        y_f = scale_value(targets[target_name][1], REF_H, self.resolutionY)
        x_i = round_coord(x_f)
        y_i = round_coord(y_f)
        return x_i, y_i, x_f, y_f  # kembalikan juga nilai float untuk logging
    
    def tap(self, target_name: str):
        x_i, y_i, *_ = self._compute_scaled_target(target_name)
        self.driver.execute_script("mobile: clickGesture", {"x": x_i, "y": y_i})

    def enter(self):
        self.driver.press_keycode(66)  # KEYCODE_ENTER

    def wait(self, sec: float):
        time.sleep(sec)

    def _screenshot_bgr(self) -> np.ndarray:
        """Screenshot → ndarray BGR (untuk OpenCV)."""
        png = self.driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png)).convert("RGB")
        return np.array(img)[:, :, ::-1]  # RGB → BGR

    def _multiscale_match(self, haystack_bgr: np.ndarray, needle_bgr: np.ndarray):
        """Kembalikan (best_score, top_left, bottom_right). None jika gagal."""
        best = (-1.0, None, None)
        for s in SCALES:
            tw = max(5, int(needle_bgr.shape[1] * s))
            th = max(5, int(needle_bgr.shape[0] * s))
            templ = cv2.resize(needle_bgr, (tw, th), interpolation=cv2.INTER_AREA)

            res = cv2.matchTemplate(haystack_bgr, templ, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best[0]:
                tl = max_loc
                br = (tl[0] + templ.shape[1], tl[1] + templ.shape[0])
                best = (max_val, tl, br)

        return best  # (score, tl, br)

    def _verify_product(self, name: str, threshold: float = MATCH_THRESHOLD):
        """True jika template produk ditemukan ≥ threshold."""
        path = PRODUCT_TEMPLATES.get(name.lower())
        if not path or not os.path.exists(path):
            print(f"[VERIFY] Template '{name}' tidak ditemukan: {path}")
            return False, 0.0, None

        screen = self._screenshot_bgr()
        templ = cv2.imread(path, cv2.IMREAD_COLOR)
        if templ is None:
            print(f"[VERIFY] Gagal load template: {path}")
            return False, 0.0, None

        score, tl, br = self._multiscale_match(screen, templ)
        ok = score >= threshold and tl is not None
        center = None
        if ok:
            cx = (tl[0] + br[0]) // 2
            cy = (tl[1] + br[1]) // 2
            center = (cx, cy)

        print(f"[VERIFY] {name} score={score:.3f} ok={ok} center={center}")
        return ok, score, center

    def change_card(self, produk: str):
        print(f"[CHANGE CARDS] Ganti kartu ke produk: {produk}")
        delay = 1.5
        steps = [
            ("tap",   "tukar_kartu"),
            ("wait",  delay),

            ("tap",   produk),
            ("wait",  delay),

            ("tap",   "tentukan_kartu"),
            ("wait",  delay),
        ]
        
        for action, value in steps:
            if action == "tap":
                self.tap(value)
            elif action == "wait":
                self.wait(value)
            else:
                raise ValueError(f"Aksi tidak dikenal: {action}")

    def tap_tukar(self):
        """
        Coba klik tombol 'Tukar' via template 'tukar.png'.
        Jika gagal, fallback ke koordinat ter-skala ('tukar_icon').
        """
        ok, score, center = False, 0.0, None
        if self.img:
            ok, score, center = self.img.tap_image("tukar", retries=3, delay=0.5)
            print(f"[TUKAR] image-tap ok={ok} score={score:.3f} center={center}")

        if not ok:
            # fallback (agar tetap jalan meski template miss)
            print(f"[TUKAR] fallback ke koordinat karena score={score:.3f}")
            self.tap("tukar_icon")

    def test_find_target(self) -> None:
        global automation_running
        global user_id_global
        global produk_global
        user_id = user_id_global if user_id_global else "123456789"
        produk_expected = produk_global if produk_global else "zeus"
        automation_running = True

        while automation_running:
            delay = 1.5
            steps = [
                ("tap_tukar", None),
                ("wait",  delay),

                # ✅ Verifikasi produk di dialog (pakai screenshot + template matching)
                ("verify", produk_expected),

                ("tap",   "input_id_teman"),
                ("type",  user_id),
                ("enter", None),
                ("wait",  delay),

                ("tap",   "tombol_cari"),
                ("wait",  delay),

                ("tap",   "tombol_tukar"),
                ("wait",  delay),

                ("tap",   "input_sandi"),
                ("type",  "10sama"),
                ("enter", None),
                ("wait",  delay),

                ("tap",   "tombol_tentukan"),
                ("wait",  delay),
            ]

            for action, value in steps:
                if action == "tap":
                    self.tap(value)
                elif action == "tap_tukar":                 # >>>
                    self.tap_tukar()
                elif action == "type":
                    self.type_text_via_keycode(value)
                elif action == "enter":
                    self.enter()
                elif action == "wait":
                    self.wait(value)
                elif action == "verify":
                    ok, score, center = self._verify_product(value)
                    if not ok:
                        # Kalau tidak cocok, kamu bisa: break / continue / raise
                        self.change_card(value)
                else:
                    raise ValueError(f"Aksi tidak dikenal: {action}")

            self.assertTrue(True)
            automation_running = False

if __name__ == '__main__':
    unittest.main()