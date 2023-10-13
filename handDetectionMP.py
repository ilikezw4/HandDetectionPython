import mediapipe as mp
import cv2
import numpy as np
from collections import deque
import re

# Constants for detection precision
MOVEMENT_THRESHOLD = 0.03
MAX_POSITIONS = 18

'''
###############################################################################
Function for filtering coordinates

arg: list of 21 coordinates 
return: string of filtered coordinates 
###############################################################################
'''


def filterNumbers(toFilterObject):
    # regex filter
    filteredList = re.findall(r'[-+]?(?:\.\d+|\d+(?:\.\d*)?)(?:[Ee][-+]?\d+)?', str(toFilterObject))
    # list comprehension + removing coma
    filteredList = [str(round(float(num) * 1000)) if 'e' not in num.lower() else '0' for num in filteredList]
    filteredString = ",".join(filteredList)

    return filteredString


'''
###############################################################################
Function for calculating the average of coordinates within a dequeue

arg: dequeue of coordinates
return: average of coordinates 
###############################################################################
'''


def calculateAverage(points):
    # calculates average
    return np.mean(points, axis=0)


'''
###############################################################################
Main Function 
###############################################################################
'''


def HandDetectionMP():
    # set up mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # saving yes/no
    doWrite = True
    # setup hand detection
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        cap = cv2.VideoCapture(0)
        file = open('./Data/data', 'w')
        # setup dequeue
        previous_hand_positions = deque(maxlen=MAX_POSITIONS)
        while cap.isOpened():
            ret, frame = cap.read()
            # resize window
            resizedFrame = cv2.resize(frame, (720, 720), interpolation=cv2.INTER_CUBIC)
            if not ret:
                break

            # grey scale
            image = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            # calc hand landmarks
            results = hands.process(image)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0].landmark
                # add hand landmarks to np array
                hand_positions = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                previous_hand_positions.append(hand_positions)

                if len(previous_hand_positions) == MAX_POSITIONS:
                    # calc average if all dequeue positions are filled
                    average_hand_position = calculateAverage(previous_hand_positions)
                    # calc difference between average and current coords
                    difference = np.linalg.norm(hand_positions - average_hand_position)
                    if difference < MOVEMENT_THRESHOLD and doWrite:
                        # print("capture")
                        data = filterNumbers(results.multi_hand_landmarks)
                        print(data)
                        file.write(str(data))
                        file.write('\n')
            # revert to colored image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # draws landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    )
            # flips image
            flippedImage = cv2.flip(image, 1)
            # shows live feed
            cv2.imshow('pretty cool dude ----v', flippedImage)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


HandDetectionMP()
