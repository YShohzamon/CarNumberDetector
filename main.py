import cv2
import easyocr
import os

# 1. OCR modelini sozlash
reader = easyocr.Reader(['en'])

# 2. Rasm yo'li
image_path = '973870.jpg'
output_path = 'result_detected.jpg'

if not os.path.exists(image_path):
    print("Rasm topilmadi!")
else:
    image = cv2.imread(image_path)
    # Tasvirni kulrang qilish (OCR aniqligini oshirishi mumkin)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Matnni aniqlash
    results = reader.readtext(gray)

    for (bbox, text, prob) in results:
        if prob > 0.5:
            print(f"Topildi: {text} | Ishonch: {prob:.2f}")
            
            # Koordinatalarni olish
            (top_left, top_right, bottom_right, bottom_left) = bbox
            tl = tuple(map(int, top_left))
            br = tuple(map(int, bottom_right))
            
            # Rasmga chizish
            cv2.rectangle(image, tl, br, (0, 255, 0), 3)
            cv2.putText(image, text, (tl[0], tl[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 4. Natijani faylga saqlash (plt.show o'rniga)
    cv2.imwrite(output_path, image)
    print(f"Natija saqlandi: {output_path}")