#Program śledzi stan oczu usera i klika enter podczas mrugnięcia
import cv2
import dlib
import numpy as np
import time
import pyautogui

class EyeDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_68.dat')
        self.left = [36, 37, 38, 39, 40, 41]
        self.right = [42, 43, 44, 45, 46, 47]
        self.kernel = np.ones((9, 9), np.uint8)
        self.threshold = 0
        self.is_eye_closed = False
        self.eye_closed_time = None
        self.enter_pressed = False

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def eye_on_mask(self, mask, side):
        points = [self.shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        mask = cv2.fillConvexPoly(mask, points, 255)
        return mask

    def contouring(self, thresh, mid, img, right=False):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        except:
            pass

    def detect(self):
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        thresh = img.copy()
        cv2.namedWindow('image')

        def nothing(x):
            pass
        cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

        while True:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)
            for rect in rects:
                shape = self.predictor(gray, rect)
                self.shape = self.shape_to_np(shape)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask = self.eye_on_mask(mask, self.left)
                mask = self.eye_on_mask(mask, self.right)
                mask = cv2.dilate(mask, self.kernel, 5)
                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = (self.shape[42][0] + self.shape[39][0]) // 2
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                self.threshold = cv2.getTrackbarPos('threshold', 'image')
                _, thresh = cv2.threshold(eyes_gray, self.threshold, 255, cv2.THRESH_BINARY)
                thresh = cv2.erode(thresh, None, iterations=2)
                thresh = cv2.dilate(thresh, None, iterations=4)
                thresh = cv2.medianBlur(thresh, 3)
                thresh = cv2.bitwise_not(thresh)
                self.contouring(thresh[:, 0:mid], mid, img)
                self.contouring(thresh[:, mid:], mid, img, True)

                # Sprawdzanie mrugania
                if np.mean(thresh) < self.threshold:
                    if not self.is_eye_closed:
                        self.eye_closed_time = time.time()
                    self.is_eye_closed = True
                else:
                    self.is_eye_closed = False
                    self.eye_closed_time = None

                # Wciskanie enter jeżeli oczy były zamknięte przez 1 s
                if self.is_eye_closed and self.eye_closed_time is not None:
                    if time.time() - self.eye_closed_time >= 1.0:
                        if not self.enter_pressed:
                            self.enter()
                            self.enter_pressed = True
                else:
                    self.enter_pressed = False

            cv2.imshow('eyes', img)
            cv2.imshow("image", thresh)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def enter(self):
        pyautogui.press('enter')