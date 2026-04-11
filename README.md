# 🚗 Car Number Detector (ANPR)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Bu loyiha avtomobil raqamlarini aniqlash va o'qish uchun yaratilgan AI tizimidir.

---

## 🚀 Loyiha Haqida
Loyiha ikki bosqichda ishlaydi:
1. **Detection:** YOLOv8 yordamida raqamni topish.
2. **Recognition:** EasyOCR yordamida matnga aylantirish.

### 📊 Ko'rsatkichlar (Training Metrics)
O'qitish jarayoni yakunida quyidagi aniqlik natijalari olindi:
* **mAP50:** `0.853` (Umumiy)
* **mAP50 (Raqamlar):** `0.907` (Juda yuqori aniqlik)
* **Tezlik:** `15.5ms` (Har bir rasm uchun)

---

## 🛠 O'rnatish va Ishga tushirish

```bash
# Loyihani klonlash
git clone [https://github.com/yshohzamon/CarNumberDetector.git](https://github.com/yshohzamon/CarNumberDetector.git)

# Kerakli kutubxonalarni o'rnatish
pip install ultralytics easyocr opencv-python
