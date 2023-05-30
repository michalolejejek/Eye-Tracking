from eyetracking import EyeDetector
from gesture import GestureDetector
import cv2

class MainApp:
    def __init__(self):
        self.eye_detector = EyeDetector()
        self.gesture_detector = GestureDetector()

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('image')

        while True:
            ret, img = cap.read()

            img = self.eye_detector.detect(img)
            img = self.gesture_detector.detect(img)

            cv2.imshow('image', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = MainApp()
    app.run()