import cv2
import easyocr
from ultralytics import YOLO
import os

# 1. To'g'ri papkani ko'rsatamiz (Logingizga qarab)
# Eslatma: Agar papka raqami o'zgargan bo'lsa, pastdagi yo'lni to'g'rilang
model_path = 'runs/detect/car_number_model4/weights/best.pt'

if not os.path.exists(model_path):
    print("Xato: Model fayli topilmadi! Iltimos runs/detect/... ichini tekshiring.")
else:
    model = YOLO(model_path)
    reader = easyocr.Reader(['en'])
    image_path = 'data/Uzbekistan/photo_5_2026-04-11_14-25-44.jpg'

    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        results = model(img)

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                if res.names[cls] == 'license-plate':
                    plate_crop = img[y1:y2, x1:x2]
                    ocr_result = reader.readtext(plate_crop)
                    plate_text = ""
                    if ocr_result:
                        plate_text = ocr_result[0][1]

                    print(f"Topildi: {plate_text} (Ishonch: {conf:.2f})")

                    # Vizualizatsiya
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, plate_text.upper(), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite('final_test_result.jpg', img)
        print("Natija 'final_test_result.jpg' fayliga saqlandi.")
    else:
        print(f"Sinov uchun rasm topilmadi: {image_path}")