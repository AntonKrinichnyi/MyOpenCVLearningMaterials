import cv2
import numpy as np
import matplotlib as plt


"""
    Фільтр середнього геометричного - це фільтрація зображення, призначена
для зглажування і зменшення шуму. Він базується на математичному середньому
геометричному. Фільтр середнього геометричного найбільш широко застосовується
для фільтрації гауссового шуму. В цілому, він допомогає згладити зображення
з найменшими втратами данних, ніж фільтр середнього арифметичного.
    Кожний піксель вихідного зображення в точці (х, у) задається добутком
пікселів в масці середнього геометричного, зведеним у ступінь 1 / m*n
    Наприклад, при використанні маски розміром 3 на 3 пісксель (х, у)
в вихідному зображенні буде добутком S(x, y) і всіх 8 оточуючих его пикселів
в степені 1/9.
    Використовуючи вихідне зображення з пікселем (х, у) в центрі:
                        5   16  22
                        6   3   18
                        12  3   15
    Маємо результат: (5*16*22*6*3*18*12*3*15)^(1/9) = 8.77
    Програма перетворює кольорове зображення з допомогою алгоритму 
середньогеометричної фільтрації. Зображення розділяється по кольорам RGB,
і з кожним отриманим одноканальним масивом відбувається претворення по формулі
середньогеометричної фільтрації. Отримані перетворені масиви об'єднуються та
виводяться
"""

def geometric_mean_algorithm(img):
    new_img = np.zeros(img.shape)
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] -1):
            roi = img[i - 1 : i + 2, j - 1 : j + 2]
            roi = roi.astype(np.float64)
            p = np.prod(roi)
            new_img[i - 1, j - 1] = p ** (1 / (roi.shape[0] * roi.shape[1]))
    return new_img.astype(np.uint8)

def rgb_geometric_mean(img):
    r, g, b = cv2.split(img)
    r = geometric_mean_algorithm(r)
    g = geometric_mean_algorithm(g)
    b = geometric_mean_algorithm(b)
    return cv2.merge([r, g, b])


if __name__ == "__main__":
    image = cv2.imread("static/noise.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.imshow("Original image", image)

    filtered_image = rgb_geometric_mean(image)

    cv2.imshow("Geometric mean filter", filtered_image)
    cv2.waitKey(0)
