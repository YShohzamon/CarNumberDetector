🚗 CarNumberDetector (ANPR System)
Ushbu loyiha YOLOv8 va OCR (Optical Character Recognition) texnologiyalari yordamida avtomobil raqamlarini aniqlash va ularni matn ko'rinishiga o'tkazish uchun ishlab chiqilgan. Loyiha real vaqt rejimida hamda statik rasmlarda yuqori aniqlik bilan ishlaydi.

✨ Xususiyatlari
YOLOv8 Detection: Avtomobil va uning davlat raqamini kadr ichidan aniqlash.

EasyOCR Integration: Aniqlangan raqamni lotin alifbosidagi matnga o'tkazish.

Custom Trained: Shaxsiy dataset asosida o'qitilgan (50 epoch).

High Performance: NVIDIA GPU (CUDA) yordamida tezkor ishlash.

Global Compatibility: Turli davlatlar (jumladan O'zbekiston) raqamlarini aniqlash imkoniyati.

📊 Model Natijalari (Training Metrics)
Model o'qitish jarayonida quyidagi ko'rsatkichlarga erishdi:

mAP50 (License Plate): 0.907 (90.7%)

Box Precision: 0.862

Inference Speed: ~15.5ms (per image)

🛠 O'rnatish
Loyiha ishlashi uchun Python 3.10+ talab qilinadi.

Loyihani klonlash:

Bash
git clone https://github.com/yshohzamon/CarNumberDetector.git
cd CarNumberDetector
Virtual muhit yaratish va kutubxonalarni o'rnatish:

Bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install ultralytics easyocr opencv-python pandas
🚀 Foydalanish
Raqamni aniqlash va o'qish uchun main.py skriptini ishga tushiring:

Python
from ultralytics import YOLO
import cv2

# Modelni yuklash
model = YOLO('runs/detect/car_number_model4/weights/best.pt')

# Rasmda aniqlash
results = model('test_image.jpg')

# Natijalarni ko'rish
for r in results:
    r.show()
📁 Papkalar Strukturasi
Plaintext
.
├── forYolov8/          # Dataset (Train, Val, Test)
├── runs/               # O'qitilgan model natijalari (weights)
├── main.py             # Asosiy dastur kodi
├── data.yaml           # Dataset konfiguratsiyasi
└── README.md           # Loyiha haqida ma'lumot
🛠 Texnologiyalar
Python

PyTorch (Deep Learning framework)

Ultralytics YOLOv8 (Object Detection)

EasyOCR (Text Recognition)

OpenCV (Image Processing)
