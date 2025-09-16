"""
    Програмні засоби обробки потоків (flows) інформації з фотоприйомників
мають декілька особливостей, що пов'язано, перш за все, с 2 факторами:
безперервністю обробки та обмеженістю ресурсів. Перший фактор визначє
зв'язок часу, яке виділяється на обробку, з просторово-часовою структурою
сигнала. Другий фактор пов'язаний з тим, що в реальних системах обмежені
можливості в продуктивності процессорів, паралельної обробки і об'єму
пам'яті. Завжди вирішується питання про оптимальність вибору алгоритму
обробки при вказаних вище обмеженнях.
    Розглянемо задачу фільтрації зображення від шумів, з використанням
фільтру Вінера.
    Фільтер Вінера називають також лінійним оптимальним фільтром, оскільки
меньше значення середньоквадратичної помилки, ніж фільтрі Вінера, в будь-якому
лінійному фільтрі отримати неможливо.
    На вхід фільтр отримує два сигнали: x[k] і d[k]. При цьому d[k] має два
компоненти - корисний сигнал s[k], який не корилюється з x[k] і шумову частину
n[k], корельовану з x[k]. Фільтр Вінера повинен мати таку частотну
характеристику, яка забезпечує на виході оптимальну в середньоквадратичному
сенсі оцінку y[k] коррельованої частини сигналу (шуму) n[k]. Ця оцінка
віднімається від d[k] і вихід (помилка) фільтра e[k] - це найкраща по 
середньоквадратичному критерію оцінка корисного сигнала. Таким чином
фільтр Вінера забезпечує оптимальну оцінку корисного сигналу, змішаного з
аддитивним шумом, по критерію мінімуму середньоквадратичної помилки.
    Вважається що фільтр є KIX-фільтром с L-го порядку (з L-коефіцієнтами).
При цьому його вихід визначається так:
                y[k] = (W**T)*X[k]
    Де W = [w0, w1,...,wL-1]**T вектор коефіцієнтів фільтру;
    X[k] = [x[k], x[k-1],...,x[k-(L-1)]]**T - вектор вхідного сигналу.
    Для оптимальної фільтрації необхідно знайти оптимальний вектор
коефіцієнтів W*. Для цього необхідно знайти функцію середньоквадратичної
похибки, взяти її похідну та прирівняти до нуля:
        e[k] = d[k] - y[k]
        e[k] = d[k] - (W**T)*X[k]*(X**T)[k]*W - 2d[k]*(X**T)[k]*W
        E[(e**2)[k]] = E[(d**2)[k]] + (W**T)*E[X[k]*(X**T)[k]]*W - 2E[d[k]*(X**T)[k]]*W
        Функція СКП
"""
import cv2
import numpy as np
from scipy.signal import wiener


def add_noise(img, sigma):
    gauss = np.random.normal(0, sigma, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, gauss)
    return noisy_img

def blur(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blurred_img = cv2.filter2D(img, -1, kernel)
    return blurred_img

if __name__ == "__main__":
    image = cv2.imread("static/someimage.jpg")
    
    if image is None:
        print("Error: Image not found.")
    else:
        # Додаємо шум та розмиття
        noisy_and_blurred_image = add_noise(image, 20)
        noisy_and_blurred_image = blur(noisy_and_blurred_image, 3)
        cv2.imshow("Noisy and Blurred Image", noisy_and_blurred_image)
        
        # Розділяємо зображення на каналы: B, G, R
        b, g, r = cv2.split(noisy_and_blurred_image)
        
        # Застосовуємо фільтр вінера до кожного каналу конвертуючи канали в float
        b_filtered = wiener(b.astype(float), (5, 5))
        g_filtered = wiener(g.astype(float), (5, 5))
        r_filtered = wiener(r.astype(float), (5, 5))
        
        # Нормалізуємо значення та переводимо назад в uint8
        b_filtered_uint8 = cv2.normalize(b_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        g_filtered_uint8 = cv2.normalize(g_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        r_filtered_uint8 = cv2.normalize(r_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # З'єднуємо канали назад в одне зображення
        filtered_image = cv2.merge([b_filtered_uint8, g_filtered_uint8, r_filtered_uint8])
        
        cv2.imshow("Wiener Filtered Image (Color)", filtered_image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
