from ultralytics import YOLO
import os


def main():
    # 1. Modelni tanlash (YOLOv8 Nano - tezkor va samarali)
    # Agar sizda avvaldan o'qitilgan model bo'lsa, uni davom ettirish mumkin
    model = YOLO('yolov8n.pt')

    # 2. Modelni o'qitish (Fine-Tuning)
    # data: data.yaml faylingizga yo'l
    # epochs: datasetni necha marta aylanib chiqishi (kamida 50 tavsiya etiladi)
    # imgsz: rasmlar o'lchami (datasetingizga mos 640 piksel)
    # device: 'cpu' (agarda GPU driveringiz eski bo'lsa) yoki 0 (agarda CUDA ishlasa)

    print("--- O'qitish boshlanmoqda ---")
    model.train(
        data='forYolov8/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # GPU xatoligi bo'lgani uchun hozircha 'cpu'
        project='runs/detect',
        name='car_number_model'
    )

    print("--- O'qitish yakunlandi ---")

    # 3. Eng yaxshi modelni yuklash
    # O'qitish tugagach, eng yaxshi vaznlar (weights) mana shu yerda saqlanadi:
    best_model_path = 'runs/detect/car_number_model/weights/best.pt'
    my_trained_model = YOLO(best_model_path)

    # 4. Sinov (Inference)
    # Bitta rasmda natijani tekshirib ko'ramiz
    test_image = 'forYolov8/test/images/some_image.jpg'  # Test papkangizdagi biror rasm nomi

    if os.path.exists(test_image):
        results = my_trained_model(test_image)

        # Natijalarni ko'rsatish
        for r in results:
            r.show()  # Rasmni ekranda ko'rsatadi
            r.save(filename='detection_result.jpg')  # Natijani saqlaydi
            print(f"Natija saqlandi: detection_result.jpg")
    else:
        print(f"Sinov uchun rasm topilmadi: {test_image}")


if __name__ == '__main__':
    main()