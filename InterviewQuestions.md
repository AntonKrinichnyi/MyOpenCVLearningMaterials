#### Основи
1. **Що таке зображення в OpenCV і як воно представлено. Наведіть приклад завантаження зображення.**
    Зображення в OpenCV це NumPy массив в якому кожен піксель представлений в вигляді значення інтенсивності (відтінки сірого), або каналами RGB/BGR (кольори). Використовується для всіх задача комп'ютерного зору, наприклад розпізнавання об'єктів.
    ```python
   import cv2
   img = cv2.imread('image.jpg')  # Load image as BGR
   ```

2. **Як отримати доступ до пікселів в зображенні OpenCV та змінити їх? Наведіть приклад**
    Доступ до значень пікселів виконаний через індексацію масивів в NumPy, що може бути корисно для попередньої обробки зображення. Наприклад зміни яскравості.
    ```python
   import cv2
   img = cv2.imread('image.jpg')
   pixel = img[100, 100]  # Access BGR at (100, 100)
   img[100, 100] = [0, 255, 0]  # Set to green
   ```

3. **В чому різниця між зображенням у відтінках сірого та кольоровими зображеннями в OpenCV?**
    Зображення в відтінках сірого мають один канал(інтенсивність), в той час як кольорові зображення (BGR в OpenCV) мають три канали. Зображення в сірих відтінках використовуються для більш простих задач, таких як виділення контурів
    ```python
   import cv2
   img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)  # Grayscale
   img_color = cv2.imread('image.jpg')  # BGR
   ```

4. **Поясніть, як зберегти зображення в OpenCV.**
    Збереження зображення дозволя зберегти оброблені результати(наприклад, відфільтроване зображення) для подальшого аналізу або вводу в модель.
    ```python
   import cv2
   img = cv2.imread('image.jpg')
   cv2.imwrite('output.jpg', img)
   ```

5. **Як перевести зображення з одного колірного простору в інший(наприклад, з BGR в відтінки сірого або HSV)? Наведіть приклад.**
    Переведення колірного простору є критично важливим для таких завдань як, сегментація на основі кольру (HSV) або спрощеня обробки (grayscale).
     ```python
   import cv2
   img = cv2.imread('image.jpg')
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   ```

6. **Напишіть функцію для зміни розміру зображення в OpenCV зі збереженням співвідношення сторін.**
    Зміна розміру важлива для підготовки зображень до обробки у нейронній мережі з фіксованими розмірами вхідних данних.
    ```python
   import cv2
   def resize_image(img, width):
       ratio = width / img.shape[1]
       height = int(img.shape[0] * ratio)
       return cv2.resize(img, (width, height))
   img = cv2.imread('image.jpg')
   resized = resize_image(img, 300)
   ```

7. **Як обрізати зображення в OpenCV? Наведіть приклади**
    Обрізання виділяє області інтересу (ROI) для цілеспрямованого аналізу, наприклад, розпізнавання обличь.
    ```python
   import cv2
   img = cv2.imread('image.jpg')
   cropped = img[50:150, 100:200]  # Crop (y1:y2, x1:x2)
   ```

8. **Поясніть як працюють канали зображення, як їх розділяти та поєднувати в OpenCV.**
    Канали (наприклад B, G, R) предстваляють кольорові компоненти, розділені для окремого аналізу або з'єднані для реконструкції зображень.
    ```python
   import cv2
   img = cv2.imread('image.jpg')
   b, g, r = cv2.split(img)  # Split channels
   merged = cv2.merge([b, g, r])  # Merge back
   ```

9. **Напиши функцію для нормалізаціхґї значень пікселів на зображенні.**
    Нормалізація масштабує значення пікселів (наприклад від 0 до 1) для узгодженого введення в моделі машинного навчання.
    ```python
   import cv2
   import numpy as np
   def normalize_image(img):
       return img / 255.0
   img = cv2.imread('image.jpg')
   normalized = normalize_image(img)
   ```

10. **Реалізуйте функію для повороту зображення без обрізання за допомогою OpenCV.**
    Обертання використовується в доповненні данних для навчання надійних моделей зору.
    ```python
    import cv2
    import numpy as np
    def rotate_image(img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(img, M, (new_w, new_h))
    img = cv2.imread('image.jpg')
    rotated = rotate_image(img, 45)
    ```

11. **Що таке порогове значення зображення в OpenCV. Наведіть приклад.**
    Порогове значення перетворює кольрове або grayscale зображення в двійкове зображення(наприклад для сегментації об'єктів).
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ```

12. **Як застсувати гаусове розмиття до щображення в OpenCV? Наведіть приклад використання.**
    Гаусове розмиття зменшує шум що сприяє виявленню ознак.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    ```

13. **Яке призначення функції `cv2.imshow` і як вона використовується?**
    Відображає зображення для налагодження або візуалізації під час розробки.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

14. **Поясніть адаптивне порогове регулювання та наведіть приклад.**
    Адаптивне порогове регулювання налаштовує пороги локально, що ідеально підходить для нерівномірного освітлення в зображеннях.
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ```

15. **Напишіть функцію для застосування фільтра підвищення різкості до зображення**
    Підвищення різкості покращує краї для кращого виявлення ознак.
    ```python
    import cv2
    import numpy as np
    def sharpen_image(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    img = cv2.imread('image.jpg')
    sharpened = sharpen_image(img)
    ```

16. **Як виконати вирівнювання гістограм в OpenCV? Наведіть приклад.**
    Вирівнювання гістограм покращує контрастність, що корисно для зображень зі слабким освітленням.
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    equalized = cv2.equalizeHist(img)
    ```

17. **Реалізуйте функцію для застосування власного ядра згортки.**
    Власні ядра дозволяють спеціалізовану філтрацію (наприклад виявлення країв)
    ```python
    import cv2
    import numpy as np
    def apply_kernel(img, kernel):
        return cv2.filter2D(img, -1, kernel)
    img = cv2.imread('image.jpg')
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered = apply_kernel(img, kernel)
    ```

18. **Напишіть функцію для виконання морфологічних операцій (наприклад розширення).**
    Морфологічні операції, такі як розширення, покращують форми для сегментації.
    ```python
    import cv2
    import numpy as np
    def dilate_image(img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    dilated = dilate_image(img)
    ```

19. **Поясніть, як обробляти шум заображення за допомогою медіанної фільтрації в OpenCV.**
    Медіанна фільтрація прибирає шум типу сіль-перець, зберігаючи при цьому краї.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    denoised = cv2.medianBlur(img, 5)
    ```

### Виявлення ознак

20. **Що таке виявленя країв (edge detection) в OpenCV, як воно виконується за допомогою Canny?**
    Виявлення країв розпізнає межі, що є вирішальним для розпізнавання об'єкту.
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    ```

21. **Як виявити кути на зображенні за домогою OpenCV? Наведіть приклади використання.**
    Виявлення кутів (наприклад Harris) визначає ключові точки для відстеження або зіставлення.
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    corners = cv2.cornerHarris(img, 2, 3, 0.04)
    ```

22. **Що таке контури в OpenCV та як їх знайти?**
    Контури - це криві, що з'єднують безперервні точки та використовуються для аналізу форм об'єктів.
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ```

23. **Напишіть функцію для виявлення та малювання контурів на зображенні.**
    Візуалізація меж об'єктів для завдань сегментації.
    ```python
    import cv2
    def draw_contours(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
    img = cv2.imread('image.jpg')
    contoured = draw_contours(img)
    ```

24. **Як використовувати SIFT для виялення ознак в OpenCV? Наведіть приклад.**
    SIFT виявляє ключові точки, незмінні щодо масштабу, для зітавлення зображень.
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    ```

25. **Поясніть перетворення Хофв для виявлення ліній та наведіть приклад.**
    Перетворення Хофа вияляє лінії, корисно для структурного аналізу
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    ```

26. **Реалізуйте функцію для зіставлення ознак між двома зображеннями за допомогою ORB**
    Зіставлення ознак вирівнює зображення для таких завдань як складання панорам.
    ```python
    import cv2
    def match_features(img1, img2):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)
    img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
    matches = match_features(img1, img2)
    ```

27. **Напиши функцію для виявлення кіл за допомогою перетворення Хофа**
    Виявлення кіл використовується в застосунках для виявлення зіниць очей.
    ```python
    import cv2
    import numpy as np
    def detect_circles(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        return img
    img = cv2.imread('image.jpg')
    circled = detect_circles(img)
    ```

28. **Поясніть як використовувати SURF для надійного виявлення ознак.**
    SURF швидший за SIFT та стійкий до масштабування та обертання, використовується для розпізнавання об'єктів (доступно в `opencv-conrib-python`).
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(img, None)
    ```

### Трансформіція зображень

29. **Що таке афінне перетвореня в OpenCV? Навеліть приклад.**
    Афінні перетворення (наприклад переміщання або обертання) зберігають лінії та використовуються для вирівнювання зображень
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 50], [0, 1, 20]])  # Translate
    translated = cv2.warpAffine(img, M, (w, h))
    ```

30. **Як перевернути зображення в OpenCV?**
    Перевертання корисне для доповнення данних в навчанні моделей зору.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    flipped = cv2.flip(img, 1)  # Horizontal flip
    ```

31. **Поясніть трансформіцію перспективи в OpenCV.**
    Трансформіція перспективи виправляє спотворення(наприклад для сканування документів).
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    pts1 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (400, 400))
    ```

32. **Напишіть функцію для застосування зсувного перетворення до зображення**
    Зсувні перетворення доповнюють дані або імітують спотворення.
    ```python
    import cv2
    import numpy as np
    def shear_image(img, shear_factor=0.2):
        h, w = img.shape[:2]
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h))
    img = cv2.imread('image.jpg')
    sheared = shear_image(img)
    ```

33. **Як виконується перетворення зображень в OpenCV? Наведіть приклад.**
    Претворення зміщує зображення, що використовується для доповнення данних.
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    h, w = img.shape[:2]
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    translated = cv2.warpAffine(img, M, (w, h))
    ```

34. **Реалізуйте функцію для нерівномірного масштабування зображення**
    Нерівномірне масштабування по-різному корегує розміри, що корисно для певних вхідних данних моделі.
    ```python
    import cv2
    def scale_image(img, scale_x, scale_y):
        return cv2.resize(img, None, fx=scale_x, fy=scale_y)
    img = cv2.imread('image.jpg')
    scaled = scale_image(img, 1.5, 0.8)
    ```

35. **Напишіть функцію для застосування гомографічного перетворення між двома зображеннями**
    Гомографія вирівнює зображення з різними перспективами (наприклад для зшивання).
    ```python
    import cv2
    import numpy as np
    def apply_homography(img1, img2, pts1, pts2):
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
        h, w = img2.shape[:2]
        return cv2.warpPerspective(img1, H, (w, h))
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    pts1 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts2 = np.float32([[10, 10], [310, 0], [10, 310], [310, 310]])
    warped = apply_homography(img1, img2, pts1, pts2)
    ```

36. **Поясніть як використовувати оптичний потік для відстеження руху в OpenCV?**
    Оптичний потік відстежує рух пікселів між кадрами що використовується в аналізі відео.
    ```python
    import cv2
    import numpy as np
    frame1 = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    ```

37. **Реалізуйте функцію для стабілізації відеокадрів за допомогою зіставлення ознак.**
    Стабілізація вирівнює кадри для плавнішої обробки відео.
    ```python
    import cv2
    import numpy as np
    def stabilize_frame(prev_frame, curr_frame):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(prev_frame, None)
        kp2, des2 = orb.detectAndCompute(curr_frame, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        h, w = prev_frame.shape[:2]
        return cv2.warpPerspective(curr_frame, H, (w, h))
    ```

### Обробка відео

38. **Як читати відеофайл в OpenCV? Наведіть приклад.**
    Зчитування відео дозволяє покадровий аналіз для таких завдань, як відстеження об'єктів.
    ```python
    import cv2
    cap = cv2.VideoCapture('video.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

39. **Що таке класс `cv2.VideoWriter` та як він використовується?**
    `VideoWriter` зберігає оброблені відеокадри (наприклад, для анотованих виводів).
    ```python
    import cv2
    cap = cv2.VideoCapture('video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    ```

40. **Як захопити вхідний сигнал з вебкамери в OpenCV?**
    Вхідний сигнал вебкамери використовується для програм зору в реальному часі, наприклад розпізнавання об'єктів або обличчя
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

41. **Напишіть фнкцію для вилучення кожного n-го кадру з відео.**
    Вилучення кадрів зменшує обсяг данних для ефективної обробки
    ```python
    import cv2
    def extract_frames(video_path, n):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % n == 0:
                frames.append(frame)
            count += 1
        cap.release()
        return frames
    ```

42. **Як обчислюється різниця між послідовними відеокадрами?**
    Функція різниці кадрів вияляє рух для відстеження або виявлення подій.
    ```python
    import cv2
    cap = cv2.VideoCapture('video.mp4')
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray
    cap.release()
    ```

43. **Реалізуйте функцію для перетворення відео у градації сірого (grayscale).**
    Перетворення у градації сірого спрощує обробку відео для таких задач ях виявлення країв.
    ```python
    import cv2
    def grayscale_video(input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))), isColor=False)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(gray)
        cap.release()
        out.release()
    ```

44. **Напишіть функцію для відстеження об'єкта у відео за допомогою віднімання фону.**
    Віднімання фону ізолює рухомі об'єкти для відстеження.
    ```python
    import cv2
    def track_object(video_path):
        cap = cv2.VideoCapture(video_path)
        fgbg = cv2.createBackgroundSubtractorMOG2()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fgmask = fgbg.apply(frame)
            cv2.imshow('Tracking', fgmask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    ```

45. **Реалізуйте функцію для виялення облич у відеопотоці за допомогою каскадів Хаара**
    Розпізнавання облич - це поширене завдання для комп'ютерного зору в реальному часі.
    ```python
    import cv2
    def detect_faces(video_path):
        cap = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Faces', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    ```

46. **Поясніть як використовувати модуль глибоких нейронних мереж (Deep Neural Networks (DNN)) в OpenCV для виявлення об'єктів?**
    Модуль DNN в OpenCV запускає попередньо навчені моделі (наприклад YOLO) для виявлення в реальному часі, інтегруючись з фреймворками, такими як TensorFlow
    ```python
    import cv2
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    ```

### Детекція та розпізнавання об'єктів

47. **Що таке каскад Хаара в OpenCV і як він використовується для розпізнавання облич?**
    Касади Хаара - це класифікатори для розпізнавання об'єктів таких як обличчя, за домогою певних ознак.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ```

48. **Як намалювати обмежувальні рамки (bounding boxes) навколо виявлених об'єктів? Наведіть приклад.**
    Обмежувальні рамки візуалізують об'єкти для анотації.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    boxes = [(50, 50, 100, 100)]  # (x, y, w, h)
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    ```

49. **Поясніть зіставлення шаблонів в OpenCV.**
    Зіставлення шаблонів знаходить мале зображення (Шаблон) на більшому зображенні, що використовується для розпізнавання однакових зображень.
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    ```

50. **Напишіть функцію для виконання зіставлення шаблонів та відображення результату.**
    Візуалізує збіги між областями для локалізації об'єктів.
    ```python
    import cv2
    import numpy as np
    def template_match(img, template):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, temp_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        h, w = temp_gray.shape
        cv2.rectangle(img, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 2)
        return img
    img = cv2.imread('image.jpg')
    template = cv2.imread('template.jpg')
    matched = template_match(img, template)
    ```

51. **Як використовувати дескриптори HOG для виявленя пішоходів**
    HOG (Histogram of Oriented Gradients) або гістограмма орієнтованих градієнтів вияляє ознаки для виявлення об'єктів таких як анприклад пішоходи.
     ```python
    import cv2
    img = cv2.imread('image.jpg')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(img, winStride=(8, 8))
    ```

52. **Реалізуйте функцію для виявлення очей на зображенні за допомогою каскадів Хаара.**
    Розпізнавання очей використовується в програмах аналізу обличчя.
    ```python
    import cv2
    def detect_eyes(img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        return img
    img = cv2.imread('image.jpg')
    eyed = detect_eyes(img)
    ```

53. **Напишіть функцію для виконання виявлення об'єктів YOLO за допомогою OpenCV та DNN.**
    YOLO забезпечує виявлення обє'ктів у реальному часі з високою точністю.
    ```python
    import cv2
    import numpy as np
    def yolo_detect(img, config_path, weights_path, classes_path):
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f]
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        boxes, confidences, class_ids = [], [], []
        h, w = img.shape[:2]
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img
    img = cv2.imread('image.jpg')
    detected = yolo_detect(img, 'yolov3.cfg', 'yolov3.weights', 'coco.names')
    ```

54. **Поясніть, як точно налаштувати попередньо навчену модель за допомогою OpenCV та DNN.**
    Точне налаштування адаптує моделі, такі як MobileNet, для конкретних завдань, використовуючи OpenCV для завантаження та виконання коду.
    ```python
    import cv2
    net = cv2.dnn.readNet('model.onnx')
    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (224, 224), swapRB=True)
    net.setInput(blob)
    output = net.forward()
    ```

55. **Реалізуйте функцію для відстеження об'єктів у реальному часі за допомогою трекерів OpenCV.**
    Трекери такі як CSRT відстежують об'єкти в різних кадрах відео
    ```python
    import cv2
    def track_object(video_path, bbox):
        cap = cv2.VideoCapture(video_path)
        tracker = cv2.TrackerCSRT_create()
        ret, frame = cap.read()
        tracker.init(frame, bbox)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    ```

### Сегментація зображень

56. **Що таке сегментація зображень і як вона виконується в OpenCV?**
    Сегментація розділяє зображення на області(наприклад об'єкти проти фону).
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    _, segmented = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ```

57. **Як використовувати алгоритм watershed для сегментації в OpenCV?**
    Wateshed розділяж об'єкти, що дотикаються на основі маркерів.
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    markers = cv2.watershed(img, np.zeros_like(gray, dtype=np.int32))
    ```

58. **Поясніть GrabCut для сегментаціх переднього плану та фону.**
    GrabCut ітеративно відокремлює передній план від фону за допомогою обмежувальної рамки.
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 200, 200)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    ```

59. **Напишіть функцію для виконання кластеризації за методом k-середніх для сегментації на основі кольору.**
    Метод k-середніх групує пікселі за кольором, сегментуючи зображення на області.
    ```python
    import cv2
    import numpy as np
    def kmeans_segmentation(img, k=3):
        pixels = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        return segmented.reshape(img.shape)
    img = cv2.imread('image.jpg')
    segmented = kmeans_segmentation(img)
    ```

60. **Як використовувати сегментацію на основі контурів в OpenCV?**
    Контури визначають межу об'єктів для сегментації
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, 255, -1)
    ```

61. **Реалізуйте функцію для семантичної сегментації використовуючи попередньо навчену модель**
    Семантична сегментація призначає пікселям мітки класі за допомогою моделей DNN.
    ```python
    import cv2
    import numpy as np
    def semantic_segmentation(img, model_path):
        net = cv2.dnn.readNet(model_path)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (512, 512), swapRB=True)
        net.setInput(blob)
        output = net.forward()
        return np.argmax(output[0], axis=0)
    img = cv2.imread('image.jpg')
    segmented = semantic_segmentation(img, 'model.onnx')
    ```

62. **Напишыть функцію для поєднання watershed та GrabCut для надійної сегментації**
    Поєднання методів підвищує точність для складних сцен
    ```python
    import cv2
    import numpy as np
    def combined_segmentation(img, rect):
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        markers = cv2.watershed(img, mask2.astype(np.int32))
        img[markers == -1] = [255, 0, 0]
        return img
    img = cv2.imread('image.jpg')
    rect = (50, 50, 200, 200)
    segmented = combined_segmentation(img, rect)
    ```

63. **Поясніть як використовувати DeepLab для сегментації в OpenCV.**
    DeepLab моделі забезпечують високоточну семантичну сегментацію, що вмкреується через модуль DNN в OpenCV.
    ```python
    import cv2
    net = cv2.dnn.readNet('deeplabv3.onnx')
    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (513, 513), swapRB=True)
    net.setInput(blob)
    output = net.forward()
    ```

64. **Реалізуйте функцію для сегментації екземплярів за допомогою Mask R-CNN**
    Сегментація екземплярів ідентифікує та маскує окремі об'єкти.
    ```python
    import cv2
    import numpy as np
    def mask_rcnn_segmentation(img, config_path, weights_path):
        net = cv2.dnn.readNetFromTensorflow(weights_path, config_path)
        blob = cv2.dnn.blobFromImage(img, swapRB=True)
        net.setInput(blob)
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
        return boxes, masks
    img = cv2.imread('image.jpg')
    boxes, masks = mask_rcnn_segmentation(img, 'mask_rcnn.cfg', 'mask_rcnn.weights')
    ```

### Калібрування камери
65. **Що таке калібрування камери в OpenCV та чому воно важливе?**
    Калібрування виправляє спотворення об'єктива, що є критично важливо для точної 3D-реконструкції.
    ```python
    import cv2
    import numpy as np
    objp = np.zeros((6*7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    ```

66. **Як знайти кути шахової дошки для калібрування? Наведіть приклад.**
    Кути шахової дошки забезпечують точки для обчислення параметрів камери.
    ```python
    import cv2
    img = cv2.imread('chessboard.jpg', cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(img, (7, 6), None)
    ```

67. **Поясніть функцію `cv2.calibrateCamera`.**
    Обчислює внутрішні/зовнішні параметри з точок об'єкта та точок зображення.
    ```python
    import cv2
    import numpy as np
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], img.shape[::-1], None, None)
    ```

68. **Напишіть функцію для усунення спотворень зображення за допомогою параметрів калібрування камери.**
    Функція усунення спотворень коригує вплив об'єктива для точності аналізу.
    ```python
    import cv2
    import numpy as np
    def undistort_image(img, mtx, dist):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        return undistorted[y:y+h, x:x+w]
    img = cv2.imread('image.jpg')
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2, k3])
    undistorted = undistort_image(img, mtx, dist)
    ```

69. **Як обчислюється похибка репроекції під час калібрування камери?**
    Похибка репроекції вимірює точність калібрування
    ```python
    import cv2
    import numpy as np
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Mean reprojection error:", mean_error / len(objpoints))
    ```

70. **Реалізуйте функцію для калібрування камери за допомогою кількох зображень.**
    Використовуємо кілька зображень шахової дошки для надійного калібрування.
    ```python
    import cv2
    import numpy as np
    def calibrate_camera(images, pattern_size=(7, 6)):
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objpoints, imgpoints = [], []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist
    ```

71. **Поясніть стереокалібрування в OpenCV**
    Стереокалібрування обчислює відносне положення двох камер для 3D-реконструкції.
    ```python
    import cv2
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, mtx1, dist1, mtx2, dist2, gray.shape[::-1])
    ```

72. **Напишіть функцію для 3D-реконструкції за допомогою стереозображень.**
    Реконструює 3D-точки зі стереопар. 
    ```python
    import cv2
    import numpy as np
    def reconstruct_3d(img1, img2, mtx1, dist1, mtx2, dist2, R, T):
        img1 = cv2.undistort(img1, mtx1, dist1)
        img2 = cv2.undistort(img2, mtx2, dist2)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
        Q = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, img1.shape[:2], R, T)[4]
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        return points_3d
    ```

73. **Реалізуйте функцію для оцінки пози камери за допомогою solvePnp**
    Оцінка пози визначає орієнтацію камери відносно об'єкта.
    ```python
    import cv2
    import numpy as np
    def estimate_pose(obj_points, img_points, mtx, dist):
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)
        return rvec, tvec
    ```

### Інтеграція Машинного Навчання
74. **Як готувати зображення в OpenCV для моделей машинного навчання?**
    Зображення змінюють в розмірі, нормалізуються та перетворюються на масиви для вхідних даних машинного навчання.
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    img = cv2.resize(img, (224, 224))  # Resize
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    ```

75. **Яка роль OpenCV у доповненні даних для машинного навчання?**
    OpenCV застосовує такі перетворення, як обертання, відображення та обрізання, для доповнення наборів данних.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    flipped = cv2.flip(img, 1)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    ```

76. **Як витягти ознаки із зображень для традиційних моделей машинного навчання?**
    Такі ознаки я HOG та SIFT, витягуються для моделей подібних до SVM.
    ```python
    import cv2
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    ```

77. **Напишіть функцію для генерації доповнених зображень для машинного навчання.**
    Доповнення збільшує різноманітність наборів даних.
    ```python
    import cv2
    import numpy as np
    def augment_image(img):
        augmented = []
        augmented.append(cv2.flip(img, 1))  # Horizontal flip
        augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
        h, w = img.shape[:2]
        M = np.float32([[1, 0, 50], [0, 1, 20]])
        augmented.append(cv2.warpAffine(img, M, (w, h)))  # Translate
        return augmented
    img = cv2.imread('image.jpg')
    augmented = augment_image(img)
    ```

78. **Як інтегрувати OpenCV зі scikit-learn для класифікації зображень?**
    Ми можемо видобудувати ознаки за допомогою OpenCV та навчати їх за допомогою scikit-learn.
    ```python
    import cv2
    from sklearn.svm import SVC
    img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    hog = cv2.HOGDescriptor()
    features = hog.compute(img)
    clf = SVC()
    clf.fit([features], [1])  # Example training
    ```

79. **Реалізуйте функцію для попередньої обробки зображень для CNN.**
    Ця функція готує зображення для фреймворків глибого навчання таких як TensorFlow.
    ```python
    import cv2
    import numpy as np
    def preprocess_for_cnn(img, size=(224, 224)):
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = img / 255.0  # Normalize
        return np.expand_dims(img, axis=0)
    img = cv2.imread('image.jpg')
    processed = preprocess_for_cnn(img)
    ```

80. **Напишіть функцію для використання OpenCV з опепередньо навченою моделлю TensorFlow.**
    Ця функція виконує виведення зображення за допомогою OpenCV та TensorFlow.
    ```python
    import cv2
    import tensorflow as tf
    def run_inference(img, model_path):
        model = tf.keras.models.load_model(model_path)
        img = preprocess_for_cnn(img)
        return model.predict(img)
    img = cv2.imread('image.jpg')
    predictions = run_inference(img, 'model.h5')
    ```

81. **Поясніть як використовувати OpenCV для виведення даних у реальному часі з моделями глибого навчання.**
    Модуль DNN в OpenCV дозволяє виводити данні в реальному часі з низькою затримкою.
    ```python
    import cv2
    net = cv2.dnn.readNet('model.onnx')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224, 224), swapRB=True)
        net.setInput(blob)
        output = net.forward()
        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

82. **Реалізуйте функцію для активного навчання за допомогою OpenCV.**
    Функція вибирає невизначені зразки для маркування, використовуючи прогнози моделі.
    ```python
    import cv2
    import numpy as np
    def active_learning(images, model):
        uncertainties = []
        for img in images:
            processed = preprocess_for_cnn(img)
            pred = model.predict(processed)
            uncertainty = -np.sum(pred * np.log(pred), axis=1)  # Entropy
            uncertainties.append(uncertainty)
        return np.argsort(uncertainties)[-10:]  # Top 10 uncertain
    ```

### Оптимізація продуктивності
83. **Як оптимізувати обробку зображень в OpenCV?**
    Використовувати ефективні функції (наприклад `cv2.resize` поза циклами) та менші розміри зображень.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    resized = cv2.resize(img, (100, 100))  # Faster processing
    ```

84. **Яка роль NumPy у продуктивності OpenCV?**
    OpenCV використовує NumPy для швидких операцій з масивами уникаючи повільних циклів Python.
     ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    img = np.clip(img + 50, 0, 255).astype(np.uint8)  # Brighten
    ```

85. **Як використовувати multi-threading з OpenCV для обробки відео?**
    Multi-threading обробка паралелізує обробку кадрів для програм які працюють в реальному часі.
    ```python
    import cv2
    import threading
    cap = cv2.VideoCapture('video.mp4')
    def process_frame(frame):
        return cv2.GaussianBlur(frame, (5, 5), 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t = threading.Thread(target=process_frame, args=(frame,))
        t.start()
    cap.release()
    ```

86. **Напишіть функцію для обробки великих зображень фрагментами.**
    Розбиття на фрагменти зменшує використання пам'яті для зображень високої роздільної здатності.
    ```python
    import cv2
    import numpy as np
    def process_in_chunks(img, chunk_size=1000):
        h, w = img.shape[:2]
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                chunk = img[y:y+chunk_size, x:x+chunk_size]
                chunk = cv2.GaussianBlur(chunk, (5, 5), 0)
                img[y:y+chunk_size, x:x+chunk_size] = chunk
        return img
    img = cv2.imread('image.jpg')
    processed = process_in_chunks(img)
    ```

87. **Як використовувати OpenCV з прискоренням від графічного процессора (GPU)?**
    Модуль CUDA в OpenCV прискорює операції на сумісних графічних процесорах.
     ```python
    import cv2
    img = cv2.imread('image.jpg')
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    gpu_blurred = cv2.cuda_GaussianBlur(gpu_img, (5, 5), 0)
    blurred = gpu_blurred.download()
    ```

88. **Реалізуйте функцію для паралелізації виявлення ознак на кількох зображеннях.**
    Паралельна обробка прискорює пакетні завдання.
    ```python
    import cv2
    from multiprocessing import Pool
    def detect_features(img):
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        return kp, des
    def parallel_feature_detection(images):
        with Pool() as pool:
            results = pool.map(detect_features, images)
        return results
    images = [cv2.imread(f'image{i}.jpg', cv2.IMREAD_GRAYSCALE) for i in range(1, 5)]
    features = parallel_feature_detection(images)
    ```

89. **Поясніть як оптимізувати обробку відео в реальному часі в OpenCV.**
    Використовувати низчу роздільну здатність, пропускати кадри та залучати графічний процесор (GPU) або multi-threading.
    ```python
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320, 240))  # Lower resolution
        cv2.imshow('Optimized', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

90. **Напишуть функцію для використання модуля CUDA в OpenCV для швидкої філтрації.**
    CUDA прискорює фільтрацію зображень для великої кількості наборів даних.
    ```python
    import cv2
    def cuda_filter(img):
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_filtered = cv2.cuda_bilateralFilter(gpu_img, 5, 50, 50)
        return gpu_filtered.download()
    img = cv2.imread('image.jpg')
    filtered = cuda_filter(img)
    ```

91. **Реалізуйте функцію для порівняння операцій в OpenCV.**
    Функція вимірює продуктивність для прийняття рішень щодо оптимізації.
    ```python
    import cv2
    import time
    def benchmark_operation(img, func, iterations=100):
        start = time.time()
        for _ in range(iterations):
            func(img)
        return (time.time() - start) / iterations
    img = cv2.imread('image.jpg')
    time_taken = benchmark_operation(img, lambda x: cv2.GaussianBlur(x, (5, 5), 0))
    print(f"Average time: {time_taken} seconds")
    ```

### Взаємодія з іншими бібліотеками
92. **Як використовувати OpenCV з NumPy для ефективної обробки зображень?**
    NumPy забезпечує швидкість операцій з масивами для зображень OpenCV.
     ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    img = np.clip(img * 1.5, 0, 255).astype(np.uint8)  # Increase brightness
    ```

93. **Яка роль Matplotlib у візуалізації результатів OpenCV?**
    Matplotlib відображає зображення та графіки для аналізу.
    ```python
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread('image.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.savefig('plot.png')
    ```

94. **Як інтегрувати OpenCV з Pandas для підготовки наборів данних?**
    Pandas організовує метадані зображень для конвеєрів машинного навчання.
    ```python
    import cv2
    import pandas as pd
    images = ['image1.jpg', 'image2.jpg']
    data = {'path': images, 'width': [cv2.imread(img).shape[1] for img in images]}
    df = pd.DataFrame(data)
    ```

95. **Напишіть функцію для візуалізації виявлення ребер OpenCV за допомогою Matplotlib.**
    Ця функція візуалізує ребра для аналізу.
    ```python
    import cv2
    import matplotlib.pyplot as plt
    def visualize_edges(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        plt.imshow(edges, cmap='gray')
        plt.savefig('edges.png')
    img = cv2.imread('image.jpg')
    visualize_edges(img)
    ```

96. **Як використовувати OpenCV з scikit-learn для кластеризації пікселів зображення?**
    Кластеризуємо пікселі для навчання.
    ```python
    import cv2
    from sklearn.cluster import KMeans
    img = cv2.imread('image.jpg')
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(pixels)
    segmented = kmeans.cluster_centers_[labels].reshape(img.shape).astype(np.uint8)
    ```

97. **Реалізуйте функцію для завантаження та попередньої обробки зображень за допомогою OpenCV та TensorFlow.**
    Ця функція готує зображення для глибокого навчання.
    ```python
    import cv2
    import tensorflow as tf
    def load_and_preprocess(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        return tf.convert_to_tensor(img)
    ```

98. **Напишіть функцію для поєднання OpenCV та Dlib для виявлення орієнтирів обличчя.**
    Ця функція інтегрує OpenCV та Dlib для точкового аналізу обличчя.
    ```python
    import cv2
    import dlib
    def facial_landmarks(img):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        return img
    img = cv2.imread('image.jpg')
    landmarked = facial_landmarks(img)
    ```

99. **Поясніть як використовувати OpenCV з PyTorch для розпізнавання об'єктів у реальному часі.**
    OpenCV попередньо обробляє фрейми, а PyTorch запускає модель.
    ```python
    import cv2
    import torch
    model = torch.load('model.pt')
    model.eval()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = preprocess_for_cnn(frame)
        with torch.no_grad():
            output = model(torch.tensor(img, dtype=torch.float32))
        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ```

100. **Реалізуйте функцію для створення набору даних зо допомогою OpenCV та Pandas.**
    Ця функція створює структуровані набори даних для машиного навчання.
    ```python
    import cv2
    import pandas as pd
    def create_dataset(image_paths, labels):
        data = []
        for path, label in zip(image_paths, labels):
            img = cv2.imread(path)
            h, w, c = img.shape
            data.append({'path': path, 'label': label, 'width': w, 'height': h})
        return pd.DataFrame(data)
    ```

### Обробка Помилок
101. **Як обробляти помилки завантаження файлів OpenCV?**
    Перевірити чи зображення завантажено правильно, щоб уникнути збоїв.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    if img is None:
        raise FileNotFoundError("Image not found")
    ```

102. **Що станеться, якщо передати невалідний параметр функції OpenCV?**
    OpenCV викликає винятки (наприклад `cv2.erorr`)
    ```python
    import cv2
    try:
        img = cv2.imread('image.jpg')
        cv2.resize(img, (0, 0))  # Invalid size
    except cv2.error as e:
        print("OpenCV error:", e)
    ```

103. **Як обробляти помилки захоплення відео в OpenCV?**
    Необхідна перевірка ініціалізації захоплення та зчитування кадру.
    ```python
    import cv2
    cap = cv2.VideoCapture('video.mp4')
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    ```

104. **Напишіть функція яка обробляє помилки при роботі із зображенням.**
    Дана функція забезпечує надійні конвеєри обробки.
    ```python
    import cv2
    def process_image(img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError("Image not found")
            return cv2.GaussianBlur(img, (5, 5), 0)
        except Exception as e:
            print(f"Error: {e}")
            return None
    ```

105. **Як вирішувати проблеми з пам'яттю для великих зображень в OpenCV?**
    Використовувати менші роздільні здатності або фрагментацію для керування пам'яттю.
    ```python
    import cv2
    try:
        img = cv2.imread('large_image.jpg')
        img = cv2.resize(img, (1000, 1000))  # Reduce size
    except MemoryError:
        print("Image too large")
    ```

106. **Реалізуй функцію для повторної спроби зчитування невдалих відеокадрів.**
    Повторні спроби обробляють тимчасові помилки у відеопотоках.
    ```python
    import cv2
    def read_frame_with_retry(cap, max_attempts=3):
        for _ in range(max_attempts):
            ret, frame = cap.read()
            if ret:
                return frame
        raise RuntimeError("Failed to read frame")
    ```

107. **Створіть власний клас вийнятків для помилок в OpenCV.**
    Цей клас визначає конкретні помилки для завдань OpenCV.
    ```python
    class OpenCVError(Exception):
        pass
    def process_image(img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise OpenCVError("Failed to load image")
        return img
    ```

108. **Напишіть функцію для обробки каскадних помилок у конвеєрі обробки.**
    Ця функція керує кількома потенційними збоями.
    ```python
    import cv2
    def process_pipeline(img_path):
        try:
            img = cv2.imread(img_path)
            if img- is None:
                raise OpenCVError("Image not found")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img, 100, 200)
            return edges
        except OpenCVError as e:
            print(f"Pipeline error: {e}")
            return None
        except cv2.error as e:
            print(f"OpenCV error: {e}")
            return None
    ```

109. **Поясніть як логувати помилки в OpenCV додатках.**
    Можна використати Python модуль `logging` для відстеження проблем в vision pipelines.
    ```python
    import cv2
    import logging
    logging.basicConfig(level=logging.ERROR)
    try:
        img = cv2.imread('image.jpg')
        cv2.resize(img, (0, 0))
    except cv2.error as e:
        logging.error(f"OpenCV error: {e}")
    ```

### Завдання з зірочкою
110. **Що таке inpainting зображень в OpenCV та як його використовують?**
    Inpainting відновлює пошкоджені області зображення, що може бути корисно для попередньої обробки.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
    inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    ```

111. **Як використовується зшивання зображень в OpenCV?**
    Зшивання об'єднує зображення в панорами.
    ```python
    import cv2
    images = [cv2.imread(f'image{i}.jpg') for i in range(1, 3)]
    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(images)
    ```

112. **Поясніть функцію `cv2.calcHist` для обчислення гістограм.**
    Данна функція обчислює гістограми для аналізу розподілу піксклів.
    ```python
    import cv2
    img = cv2.imread('image.jpg')
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # Blue channel
    ```

113. **Напишіть функцію для виконання домалювання зображення за допомогою динамічної маски.**
    Дана функція відновлює певні області на основі введених користувачем даних.
    ```python
    import cv2
    import numpy as np
    def inpaint_image(img, points):
        mask = np.zeros(img.shape[:2], np.uint8)
        for (x, y) in points:
            cv2.circle(mask, (x, y), 5, 255, -1)
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    img = cv2.imread('image.jpg')
    points = [(100, 100), (150, 150)]
    inpainted = inpaint_image(img, points)
    ```

114. **Як використовувати OpenCV для доповненої реальності?**
    Необхідно накладати віртуальні об'єкти за допомогою оцінки поза та перетворень.
    ```python
    import cv2
    import numpy as np
    img = cv2.imread('image.jpg')
    obj_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1,1, 0]], dtype=np.float32)
    img_points = np.array([[100, 100], [200, 100], [100, 200],[200, 200]], dtype=np.float32)
    ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx,dist)
    ```

115. **Реалізуйте функцію для обчислення та візуалізації гістограм зображення.**
    Ця функція аналізує розподіли кольорів та інтенсивності.
    ```python
    import cv2
    import matplotlib.pyplot as plt
    def plot_histogram(img):
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.savefig('histogram.png')
    img = cv2.imread('image.jpg')
    plot_histogram(img)
    ```

116. **Релазуйте функцію для доповненої реальності в реальному часі за допомгою OpenCV.**
    Ця функція накладає об'єкти на відеопотоки.
    ```python
    import cv2
    import numpy as np
    def ar_overlay(video_path, obj_img, obj_points, img_points,mtx, dist):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ret, rvec, tvec = cv2.solvePnP(obj_points,img_points, mtx, dist)
            imgpts, _ = cv2.projectPoints(np.float32([[0, 0,0]]), rvec, tvec, mtx, dist)
            cv2.warpPerspective(obj_img, frame, imgpts)
            cv2.imshow('AR', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    ```

117. **Як використовувати OpenCV для SLAM (побудова карти невідомого простору навколо себе в робототехніці)**
    ```python
    import cv2
    orb = cv2.ORB_create()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp, des = orb.detectAndCompute(frame, None)
        # Process for SLAM
    ```

118. **Реалізуйте функцію для оцінки глибини за допомогою стереобачення.**
    Оцінює глибину за парними зображеннями.
    ```python
    import cv2
    def depth_map(img1, img2):
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
        return disparity / 16.0
    img1 = cv2.imread('left.jpg')
    img2 = cv2.imread('right.jpg')
    depth = depth_map(img1, img2)
    ```

### Додаткові питання
119. **Напишіть функцію для виявлення та видалення зеленого фону екрана.**
    Функція хромакею для монтажу
    ```python
    import cv2
    import numpy as np
    def remove_green_screen(img, lower_green=(0, 100, 0),upper_green=(100, 255, 100)):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        return cv2.bitwise_and(img, img, mask=mask)
    img = cv2.imread('image.jpg')
    result = remove_green_screen(img)
    ```

120. **Реалізуйте функцію для обчислення градієнтів зображення  за допомогою операторів Собеля.**
    Функція виділяє краї градієнтів для вилучення ознак.
    ```python
    import cv2
    def compute_gradients(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        return sobelx, sobely
    img = cv2.imread('image.jpg')
    grad_x, grad_y = compute_gradients(img)
    ```

121. **Напишіть функцію для виявлення тексту на зображенні за допомгою детектора EAST.**
    Виялення тексту для програм оптичного розпізнавання символів(OCR)
    ```python
    import cv2
    import numpy as np
    def detect_text(img, model_path):
        net = cv2.dnn.readNet(model_path)
        blob = cv2.dnn.blobFromImage(img, 1.0, (320, 320), (12368, 116.78, 103.94), swapRB=True)
        net.setInput(blob)
        (scores, geometry) = net.forward(['feature_fusion/Conv_7Sigmoid', 'feature_fusion/concat_3'])
        return scores, geometry
    img = cv2.imread('image.jpg')
    scores, geometry = detect_text(img, 'east_text_detection.pb')
    ```

122. **Реалізуйте функцію для створення ефекту мозаїки на зображенні.**
    Їх використовують для анонімізації або стилізації зображення.
    ```python
    import cv2
    def mosaic_effect(img, block_size=10):
        h, w = img.shape[:2]
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = img[y:y+block_size, x:x+block_size]
                avg_color = block.mean(axis=(0, 1)).astype(npuint8)
                img[y:y+block_size, x:x+block_size] = avg_color
        return img
    img = cv2.imread('image.jpg')
    mosaiced = mosaic_effect(img)
    ```

123. **Напишіть функцію для змішування зображень за допомогою OpenCV.**
    Змішування поєднує зображення для створення ефектів або накладень.
    ```python
    import cv2
    def blend_images(img1, img2, alpha=0.5):
        return cv2.addWeighted(img1, alpha, img2, 1-alpha, 0.0)
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    blended = blend_images(img1, img2)
    ```

124. **Реалізуйте функцію для виявлення та підрахунку об'єктів на зображенні.**
    Функція підраховує сегментовані об'єкти.
    ```python
    import cv2
    import numpy as np
    def count_objects(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)
    img = cv2.imread('image.jpg')
    count = count_objects(img)
    ```