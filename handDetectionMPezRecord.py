import mediapipe as mp
import cv2
import numpy as np
from collections import deque
import re

# Constants
MOVEMENT_THRESHOLD = 0.04
MAX_POSITIONS = 15


def filterNumbers(toFilterObject):
    filteredList = re.findall(r'[-+]?(?:\.\d+|\d+(?:\.\d*)?)(?:[Ee][-+]?\d+)?', str(toFilterObject))
    filteredList = [str(float(num) * 1000) if 'e' not in num.lower() else '0' for num in filteredList]
    filteredString = ",".join(filteredList)
    return filteredString


def calculateAverage(points):
    return np.mean(points, axis=0)


def HandDetectionMP():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        cap = cv2.VideoCapture(0)
        file = open('./Data/data', 'w')
        doWrite = True
        previous_hand_positions = deque(maxlen=MAX_POSITIONS)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0].landmark
                hand_positions = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                previous_hand_positions.append(hand_positions)

                if len(previous_hand_positions) == MAX_POSITIONS:
                    average_hand_position = calculateAverage(previous_hand_positions)
                    difference = np.linalg.norm(hand_positions - average_hand_position)
                    if difference < MOVEMENT_THRESHOLD and doWrite:
                        print("capture")
                        data = filterNumbers(results.multi_hand_landmarks)
                        file.write(str(data))
                        file.write(" /// ")
                        # break

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    )

            flippedImage = cv2.flip(image, 1)
            cv2.imshow('Raw Webcam Feed', flippedImage)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


HandDetectionMP()
