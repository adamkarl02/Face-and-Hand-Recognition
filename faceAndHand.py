import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import mediapipe as mp

class GestureActions:
    def on_thumb_up(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[3]
        if thumb_tip.y < thumb_base.y:
            print("Thumb up gesture recognized.")
            return True
        return False

    def on_thumb_down(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[3]
        if thumb_tip.y > thumb_base.y:
            print("Thumb down gesture recognized.")
            return True
        return False

    def on_smile(self, face_data):
        left_mouth_corner = face_data[61]  
        right_mouth_corner = face_data[291] 
        if self.calculate_distance(left_mouth_corner, right_mouth_corner) > 50.5:
            print("Smile detected.")
            return True
        return False

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def perform_actions(self, hand_landmarks=None, face_data=None):
        if hand_landmarks:
            self.on_thumb_up(hand_landmarks) or self.on_thumb_down(hand_landmarks)
        if face_data:
            self.on_smile(face_data)
        # Note: The "or" chaining assumes that only one gesture will be detected at a time.
        # If multiple gestures need to be detected, we would need to call each method separately.


class FaceHandDetector:
    def __init__(self):
        self.face_mesh_detector = FaceMeshDetector(maxFaces=1)
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.gesture_actions = GestureActions()

    def process_frame(self):
        success, image = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            return None, None

        image = cv2.flip(image, 1)
        image, faces = self.face_mesh_detector.findFaceMesh(image, draw=True)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(image_rgb)
        return image, faces, hand_results

    def draw_hand_landmarks(self, image, hand_results):
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS)

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def detect_gestures(self, faces, hand_results):
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.gesture_actions.perform_actions(hand_landmarks=hand_landmarks)
        if faces:
            self.gesture_actions.perform_actions(face_data=faces[0])

    def run(self):
        frame_skip = 2  # process every 2nd frame
        frame_counter = 0
        while True:
            image, faces, hand_results = self.process_frame()
            if frame_counter % frame_skip == 0 and image is not None:
                self.draw_hand_landmarks(image, hand_results)
                self.detect_gestures(faces, hand_results)
                cv2.imshow('FaceMesh and Hand Tracking', image)
            frame_counter += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        self.release_resources()

if __name__ == "__main__":
    detector = FaceHandDetector()
    detector.run()
