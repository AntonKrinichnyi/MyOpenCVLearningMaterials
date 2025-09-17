"""
    Визначення переважаючих кольорів на зображенні з використанням
кластеризації за методом k-середніх

    Визначення переважаючих кольорів на зображенні це важлива робота при
роботі з палітрою кольорів. Реалізуємо програму, яка виводить зображення
та лінійку кольорів в співвідношенні один до одного. Програма буде виконувати
наступні дії:
        1. Завантажувати зображення з файла
        2. Створювати об'єкт, який перетворений з картинки зображення на
    список пікселів цієї картинки.
        3. Знаходимо кластер за методом k-середніх
        4. Візуалізуємо отриманий кластер
    
    Під час розрахунку переважаючого кольору використовується кластеризація
методом k-середніх. Кластеризація - це задача групування наборів об'єктів
таким чином, щоб об'єкти в одній групі (яку називають кластером) були більш
схожі один на одного, ніж на об'єкти в інших групах (кластерах). Для 
k-середніх кластери представлені центральним вектором, коли кількість
кластерів фіксовано на k, знайти k цетр та призначити об'єкти найближчому
центру кластера, так щоб квадрати відстаней від кластера були мінімізовані
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans

def visualize_colors(cluster, centroids):
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=labels)
    hist = hist.astype(float)
    hist /= hist.sum()

    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for percent, color in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), color.astype("uint8").tolist(), -1)
        start = end
    return rect

if __name__ == "__main__":
    image = cv2.imread("static/someimage.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    cluster = KMeans(n_clusters=5).fit(reshape)
    visualize = visualize_colors(cluster, cluster.cluster_centers_)
    visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)

    cv2.imshow("Visualize", visualize)
    cv2.waitKey()
