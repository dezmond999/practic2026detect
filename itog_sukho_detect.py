# minimal_detection_filepicker.py
# С выбором файла через диалоговое окно

from ultralytics import YOLO
import cv2
from tkinter import filedialog
import os

# 1. Загружаем модель
print("Загрузка модели...")
model = YOLO("best.pt")
print("✅ Модель загружена")

# 2. Выбираем фото
file_path = filedialog.askopenfilename(
    title="Выберите фото",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("Файл не выбран")
    exit()

print(f"Загружено: {os.path.basename(file_path)}")

# 3. Детекция
print("Детекция...")
results = model(file_path, conf=0.25)

# 4. Сохраняем результат
output_path = os.path.splitext(file_path)[0] + "_detected.jpg"
result_image = results[0].plot()
cv2.imwrite(output_path, result_image)

print(f"✅ Сохранено: {output_path}")

# 5. Показываем
cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()