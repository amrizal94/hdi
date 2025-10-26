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
    # ROI hasil Appium Inspector (di resolusi dasar kamu pas ngasih “kotak biru”)
    ROI_BIRU_BASE = (375, 165, 865, 475)  # ganti dengan angkamu
    
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
        
        # Step 1: buka dialog tukar kartu
        self.tap_image("btn_change_card")
        self.wait(delay)

        # Step 2: urutkan langkah tukar kartu
        ganti_kartu = self.tap_image("choose_card_" + produk.lower())
        self.wait(delay)
        if not ganti_kartu:
            print(f"[CHANGE CARDS] Gagal pilih kartu: {produk}")
            self.tap_image("btn_close")  # tutup dulu popup kalau ada
            return
        else:
            return True
        
        


    def tap_image(self, image_name: str = "tukar"):
        """
        Coba klik tombol 'Tukar' via template 'tukar.png'.
        Jika gagal, fallback ke koordinat ter-skala ('tukar_icon').
        """
        ok, score, center = False, 0.0, None
        if self.img:
            ok, score, center = self.img.tap_image(image_name, retries=3, delay=0.5)
            print(f"[TUKAR] image-tap = {image_name} ok={ok} score={score:.3f} center={center}")

        if not ok:
            print(f"[TUKAR] fallback ke koordinat karena score={score:.3f}")
            return
        else:
            return True

    def detect_popup(self, popup_name: str = "popup_password"):
        """
        Deteksi popup berdasarkan template.
        Return dict:
            {
                "found": True/False,
                "name": popup_name,
                "score": float,
                "rect": ((x1,y1),(x2,y2)),
                "center": (cx,cy)
            }
        """

        if not self.img:
            return {"found": False, "name": popup_name, "score": 0.0, "rect": None, "center": None}

        res = self.img.find(popup_name)
        
        return {
            "found": res.get("ok", False),
            "name": popup_name,
            "score": res.get("score", 0.0),
            "rect": res.get("rect"),
            "center": res.get("center"),
    }

    def find_user(self, is_use_ocr: bool = False, save_debug: bool = False) -> str | None:
        """
        Cari user:
        - Jika is_use_ocr=True: cek apakah ID target sudah tampil → langsung TUKAR.
        - Jika belum, lakukan input + CARI, lalu klik TUKAR.
        Return: ID yang terdeteksi (pre/post) atau None.
        """
        global user_id_global
        delay = 1.5
        user_id = user_id_global if user_id_global else "123456789"

        found_id: str | None = None

        # ---- MODE OCR (cepat, skip input bila sudah ada) ----
        if is_use_ocr and self.img:
            # PHASE A: cek dulu ROI pixel (kotak biru)
            pre_id = self.img.read_user_id_pixels(self.ROI_BIRU_BASE, save_debug=save_debug)
            print(f"[OCR] pre_id={pre_id} target={user_id}")

            if pre_id == user_id:
                # langsung klik TUKAR (pakai wrapper yang sudah ada)
                self.tap_image("tombol_tukar")
                self.wait(delay)
                return pre_id  # sudah selesai
            # else: lanjut ke fase input + cari

        # ---- PHASE B: INPUT + CARI (jalankan juga jika OCR dimatikan) ----
        # fokus kolom, ketik, enter
        self.tap("input_id_teman")
        self.type_text_via_keycode(user_id)
        self.enter()
        self.wait(delay)

        self.tap("tombol_cari")
        self.wait(delay)

        # Baca lagi dengan OCR (pakai chain yang lebih robust)
        if is_use_ocr and self.img:
            post_id = self.img.read_user_id() if self.img else None
            print(f"[OCR] post_id={post_id}")

        # klik tombol tukar di baris hasil
        self.tap("tombol_tukar")
        self.wait(delay)

        return found_id

    def restart_app(self):
        self.driver.terminate_app("com.neptune.domino")
        self.driver.activate_app("com.neptune.domino")
        self.wait(5.0)

    def test_find_target(self) -> None:
        global automation_running
        global user_id_global
        global produk_global
        user_id = user_id_global if user_id_global else "123456789"
        produk_expected = produk_global if produk_global else "zeus"
        automation_running = True

        while automation_running:
            delay = 1.5

            # 1) Buka dialog & verifikasi produk
            self.tap_image(); self.wait(delay)

            popup = self.detect_popup("popup_sesi_habis")
            if popup["found"]:
                print("[INFO] Sesi habis → Restarting app ...")
                self.restart_app()
                continue  # ulangi dari awal

            ok, score, center = self._verify_product(produk_expected)
            if not ok:
                is_changed = self.change_card(produk_expected)
                if not is_changed:
                    print("[INFO] Gagal ganti kartu → ulangi dari awal ...")
                    self.tap_image("btn_close")  # tutup dulu popup kalau ada
                    break  # ulangi dari awal

            # Step 3: konfirmasi tukar kartu
            self.tap_image("btn_confirm_card")
            self.wait(delay)

            # 4) Cari user & tukar
            self.find_user(is_use_ocr=True, save_debug=False)

            popup = self.detect_popup("popup_sesi_habis")
            if popup["found"]:
                print("[INFO] Sesi habis → Restarting app ...")
                self.restart_app()
                continue  # ulangi dari awal

            # 5) Input sandi & tentukan
            popup = self.detect_popup("popup_password")
            if popup["found"]:
                print(f"[POPUP] Detected: {popup['name']} score={popup['score']:.3f}")
                # lakukan step password
                print("[POPUP] Detected password popup")
                self.tap("input_sandi")
                self.type_text_via_keycode("10sama")
                self.enter(); self.wait(delay)

                self.tap("tombol_tentukan")
                self.wait(delay)
            else:
                print(f"[POPUP] No {popup['name']} detected")

            # 6) Konfirmasi tentukan kartu
            self.tap_image("btn_confirm_card")
            self.wait(delay)

            # Selesai satu siklus
            self.tap_image("btn_close")
            self.wait(delay)

            self.assertTrue(True)
            automation_running = False

if __name__ == '__main__':
    unittest.main()