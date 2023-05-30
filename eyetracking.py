import cv2
import dlib
import numpy as np

class EyeDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()  # Inicjalizacja detektora twarzy
        self.predictor = dlib.shape_predictor('shape_68.dat')  # Ładowanie modelu predykcyjnego do rozpoznawania punktów charakterystycznych twarzy
        self.left = [36, 37, 38, 39, 40, 41]  # Indeksy punktów reprezentujących lewe oko
        self.right = [42, 43, 44, 45, 46, 47]  # Indeksy punktów reprezentujących prawe oko
        self.kernel = np.ones((9, 9), np.uint8)  # Kernel do operacji morfologicznych
        self.threshold = 0  # Wartość progowa do binaryzacji

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)  # Inicjalizacja tablicy na współrzędne punktów charakterystycznych
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)  # Konwersja punktów charakterystycznych na współrzędne (x, y)
        return coords

    def eye_on_mask(self, mask, side):
        points = [self.shape[i] for i in side]  # Pobranie punktów charakterystycznych dla określonej strony oka
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)  # Wypełnienie poligonu na masce dla danego oka
        return mask

    def contouring(self, thresh, mid, img, right=False):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Wyszukiwanie konturów na binarnej masce
        try:
            cnt = max(cnts, key=cv2.contourArea)  # Wybór największego konturu
            M = cv2.moments(cnt)  # Obliczanie momentów konturu
            cx = int(M['m10'] / M['m00'])  # Obliczanie środka ciężkości konturu w osi X
            cy = int(M['m01'] / M['m00'])  # Obliczanie środka ciężkości konturu w osi Y
            if right:
                cx += mid  # Dostosowanie współrzędnych środka ciężkości dla prawego oka
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)  # Rysowanie okręgu w miejscu środka ciężkości
        except:
            pass

    def detect(self):
        cap = cv2.VideoCapture(0)  # Inicjalizacja kamery
        ret, img = cap.read()  # Odczyt pierwszej klatki
        thresh = img.copy()
        cv2.namedWindow('image')  # Utworzenie okna

        def nothing(x):
            pass
        cv2.createTrackbar('threshold', 'image', 0, 255, nothing)  # Utworzenie suwaka do regulacji wartości progu

        while True:
            ret, img = cap.read()  # Odczyt klatki z kamery
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Konwersja do skali szarości
            rects = self.detector(gray, 1)  # Wykrywanie twarzy na obrazie
            for rect in rects:
                shape = self.predictor(gray, rect)  # Predykcja punktów charakterystycznych twarzy
                self.shape = self.shape_to_np(shape)  # Konwersja punktów charakterystycznych na tablicę NumPy
                mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Inicjalizacja maski
                mask = self.eye_on_mask(mask, self.left)  # Wypełnienie poligonu na masce dla lewego oka
                mask = self.eye_on_mask(mask, self.right)  # Wypełnienie poligonu na masce dla prawego oka
                mask = cv2.dilate(mask, self.kernel, 5)  # Dylatacja maski
                eyes = cv2.bitwise_and(img, img, mask=mask)  # Przyciemnienie obszarów spoza oczu
                mask = (eyes == [0, 0, 0]).all(axis=2)  # Tworzenie maski dla obszarów czarnych
                eyes[mask] = [255, 255, 255]  # Ustawienie koloru białego dla obszarów czarnych
                mid = (self.shape[42][0] + self.shape[39][0]) // 2  # Obliczenie środka między oczami
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)  # Konwersja do skali szarości
                self.threshold = cv2.getTrackbarPos('threshold', 'image')  # Odczyt wartości progu z suwaka
                _, thresh = cv2.threshold(eyes_gray, self.threshold, 255, cv2.THRESH_BINARY)  # Binaryzacja obrazu oczu
                thresh = cv2.erode(thresh, None, iterations=2)  # Erozja obrazu
                thresh = cv2.dilate(thresh, None, iterations=4)  # Dylatacja obrazu
                thresh = cv2.medianBlur(thresh, 3)  # Rozmycie medianowe
                thresh = cv2.bitwise_not(thresh)  # Negacja obrazu
                self.contouring(thresh[:, 0:mid], mid, img)  # Wykrywanie konturu lewego oka
                self.contouring(thresh[:, mid:], mid, img, True)  # Wykrywanie konturu prawego oka
            cv2.imshow('eyes', img)  # Wyświetlanie obrazu z zaznaczonymi oczami
            cv2.imshow("image", thresh)  # Wyświetlanie binarnej maski oczu
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Oczekiwanie na naciśnięcie klawisza 'q' w celu zakończenia programu
                break

        cap.release()  # Zwolnienie kamery
        cv2.destroyAllWindows()  # Zamknięcie okien