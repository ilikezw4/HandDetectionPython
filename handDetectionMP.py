import mediapipe as mp
import cv2
import numpy as np
import re
from collections import deque


# Constants for detection precision
MOVEMENT_THRESHOLD = 0.05
MAX_POSITIONS = 18

'''
###############################################################################
Function for filtering coordinates

arg: list of 21 coordinates 
return: string of filtered coordinates 
###############################################################################
'''


def filter_numbers(to_filter_object):
    filtered_list = re.findall(r'[-+]?(?:\.\d+|\d+(?:\.\d*)?)(?:[Ee][-+]?\d+)?', str(to_filter_object))
    filtered_list = [str(round(float(num) * 1000)) if 'e' not in num.lower() else '0' for num in filtered_list]
    base_coord_x, base_coord_y = filtered_list[0], filtered_list[1]

    for i in range(3, len(filtered_list), 3):
        filtered_list[i] = str(int(filtered_list[i]) - int(base_coord_x))
        filtered_list[i + 1] = str(int(filtered_list[i + 1]) - int(base_coord_y))

    filtered_string = ",".join(filtered_list)
    return filtered_string


'''
###############################################################################
Function for calculating the average of coordinates within a dequeue

arg: dequeue of coordinates
return: average of coordinates 
###############################################################################
'''


def calculate_average(points):
    return np.mean(points, axis=0)


'''
###############################################################################
Main Function 
###############################################################################
'''


def hand_detection_mp():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # saving yes/no
    do_write = True

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        cap = cv2.VideoCapture(0)

        with open('./Data/Datasets/Fast/Z', 'w') as file:
            previous_hand_positions = deque(maxlen=MAX_POSITIONS)

            while cap.isOpened():
                ret, frame = cap.read()
                resized_frame = cv2.resize(frame, (720, 720), interpolation=cv2.INTER_CUBIC)

                if not ret:
                    break

                image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0].landmark
                    hand_positions = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                    previous_hand_positions.append(hand_positions)

                    if len(previous_hand_positions) == MAX_POSITIONS:
                        average_hand_position = calculate_average(previous_hand_positions)
                        difference = np.linalg.norm(hand_positions - average_hand_position)

                        if difference < MOVEMENT_THRESHOLD and do_write:
                            data = filter_numbers(results.multi_hand_landmarks)
                            print(data)
                            file.write(str(data))
                            file.write('\n')

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

                flipped_image = cv2.flip(image, 1)
                cv2.imshow('pretty cool dude ----v', flipped_image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()


hand_detection_mp()
