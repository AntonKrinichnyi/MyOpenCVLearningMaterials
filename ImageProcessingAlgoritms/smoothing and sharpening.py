"""
    Регулювання різкості підсилює чіткість меж на зображенні або навпаки,
згладжує їх. Регулювання різкості дозволяє збільшити якість більшості
зображень незалежно від того, яким чином вони отримані. Нище представлені
приклади роботи программ с застосуванням функцій OpenCV та PIL (Python Imaging
Library).
    В OpenCV для підвищення різкості використовується фільтрація. Більш високі
частоти контролюють краї, а більш низькі частоти контролюють зміст зображення.
Крах формуюються, коли є різкий перехід від одного значення пікселя до іншого,
наприклад 0 та 255 в сусідній комірці.
    Для сглажування зображення використовують два методи: bilateralFilter та 
GaussianBlur. Двосторонній фільтр - це нелінійний згладжуюючий фільтр з 
збереженням меж та зменшенням шуму для зображень. Він замінює інтенсивність
кожного пікселя середньозваженим значенням інтенсивності з сусідніх пікселів.
Гауссова фільтрація - це середньозважене значення інтенсивності сусідніх
позицій з вагою, яка зменшується з просторовою відстанню до центральної
позиції.
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def sharpened(image):
    kernel = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    cv2.imshow("Image Sharpening", sharpened)

def gaussian(image):
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), sigmaX=0)
    cv2.imshow("Gaussian blur", gaussian_blur)

def bilateral(image):
    bilateral_blur = cv2.bilateralFilter(image, 15, 80, 80)
    cv2.imshow("Bilateral blur", bilateral_blur)

def pil_work():
    image = Image.open("static/someimage.jpg")
    enchancer = ImageEnhance.Sharpness(image)

    image_sharpened = enchancer.enhance(4)
    image_sharpened.save("static/image_sharpened.jpg")

    blured_image = enchancer.enhance(0.0005)
    blured_image.save("static/blured_image.jpg")


if __name__ == "__main__":
    image = cv2.imread("static/someimage.jpg")

    cv2.imshow("Original image", image)
    sharpened(image)
    gaussian(image)
    bilateral(image)
    pil_work()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
