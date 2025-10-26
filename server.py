from fastapi import FastAPI
import threading
import unittest

# ⬇️ IMPORT Test dari main.py
import main   # pastikan main.py ada di folder yang sama

app = FastAPI()

def run_automation():
    try:
        print("[Automation] mulai...")
        # Susun dan jalankan 1 test case: test_find_target
        suite = unittest.TestSuite()
        suite.addTest(main.TestAppium("test_find_target"))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        print("[Automation] selesai. success=", result.wasSuccessful())
    finally:
        main.automation_running = False

@app.get("/api/kirim")
async def kirim(userid: str, produk: str):
    if main.automation_running:
        return {"status": False, "message": "Automation masih berjalan, mohon tunggu..."}
    
    # cek appium server ON?
    if not main.is_appium_running():
        return {"status": False, "message": "Appium server belum menyala! Tolong start Appium dulu."}

    main.user_id_global = userid
    main.produk_global = produk
    threading.Thread(target=run_automation, daemon=True).start()
    return {"status": True, "message": f"Automation dimulai untuk user {userid} produk {produk}"}

@app.get("/api/test")
async def test():
    return {"message": "OK"}
