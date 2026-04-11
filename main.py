import cv2
import easyocr
from ultralytics import YOLO
import os

# 1. To'g'ri papkani ko'rsatamiz (Logingizga qarab)
# Eslatma: Agar papka raqami o'zgargan bo'lsa, pastdagi yo'lni to'g'rilang
model_path = 'runs/detect/runs/detect/car_number_model4/weights/best.pt'

if not os.path.exists(model_path):
    print("Xato: Model fayli topilmadi! Iltimos runs/detect/... ichini tekshiring.")
else:
    # Modellarni yuklash
    model = YOLO(model_path)
    reader = easyocr.Reader(['en'])  # O'zbekiston raqamlari uchun lotincha kifoya

    # Test rasm yo'li (Birorta rasm qo'ying)
    image_path = '/home/yshohzamon/MyWorks/localGithub/AIProjects/CarNumberDetector/dataset/images/Cars7.png'

    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        results = model(img)

        for res in results:
            for box in res.boxes:
                # Koordinatalar
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                if res.names[cls] == 'license-plate':
                    # Raqamni kesib olish
                    plate_crop = img[y1:y2, x1:x2]

                    # OCR orqali o'qish
                    ocr_result = reader.readtext(plate_crop)

                    plate_text = ""
                    if ocr_result:
                        plate_text = ocr_result[0][1]  # Eng yuqori ehtimolli matn

                    print(f"Topildi: {plate_text} (Ishonch: {conf:.2f})")

                    # Vizualizatsiya
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, plate_text.upper(), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite('final_test_result.jpg', img)
        print("Natija 'final_test_result.jpg' fayliga saqlandi.")
    else:
        print(f"Sinov uchun rasm topilmadi: {image_path}")