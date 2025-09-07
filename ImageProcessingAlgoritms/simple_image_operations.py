import cv2

"""
В цьому файлі будуть вказані приклади використання функцій в OpenCV
для виконання простих операцій над зображенням. До простих операцій
відносяться: обрізка зображення, зміна розміру зображення, поворот
зображення, додавання ліній або інших фігур на зображення.
"""

image = cv2.imread("static/someimage.jpg")
cv2.imshow("Stock image", image)
cv2.waitKey(0)

def show_image(image, description): # Функція для відображення зображень
    cv2.namedWindow(description, cv2.WINDOW_NORMAL)
    cv2.imshow(description, image)
    cv2.waitKey(0)

image_copy = image.copy()
show_image(image_copy[10:480, 480:1100], "Cat image") # Обрізка зображення

image_copy2 = image.copy()
zoom = (int(image_copy2.shape[1] * 50 / 100), int(image_copy2.shape[0] * 50 / 100)) # Задаємо значення для зміни розміру зображення в %
show_image(cv2.resize(image_copy2, zoom, interpolation=cv2.INTER_AREA), "Resize on 50%") # Зменшуємо зображення

image_copy3 = image.copy()
h, w = image_copy3.shape[ :2] # Беремо значення висоти та ширини
center = (w // 2, h // 2) # Знаходимо центр зображення
show_image(cv2.warpAffine(image_copy3, cv2.getRotationMatrix2D(center, 70, 1), (w, h)), "70 degree turn") # Повертаємо зображення на 70 градусів

image_copy4 = image.copy()
cv2.line(image_copy4, (100, 100), (110, 810), (0, 255, 0), 10) # Малюємо лінію
cv2.circle(image_copy4, (200, 200), 90, (255, 0, 0), 8) # Малюємо коло
cv2.rectangle(image_copy4, (400, 400), (200, 200), (0, 0, 255), 8) # Малюємо квадрат
show_image(image_copy4, "Figures")

image_copy5 = image.copy()
cv2.putText(image_copy5, "Hello World", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 250, 200), 8) # Розміщуємо текст на зображенні
show_image(image_copy5, "Text")
