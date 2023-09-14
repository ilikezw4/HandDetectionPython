import mediapipe as mp
import cv2
import re
import numpy as np

"""
*******************************************************************************
Function in Order to Filter coordinates received
[Removes all Symbols other then Numbers (positive and negative)]
[Numbers with ' e^x ' which are smaller then ' ^-03 ' are set 0]
Returns filtered coordinate String  21 x 3 coordinates  (x, y, z per joint)  
*******************************************************************************
"""


def filterNumbers(toFilterObject):
    # Puts a Regex on the List of coordinates to Filter useless information
    filteredList = re.findall(r'[-+]?(?:\.\d+|\d+(?:\.\d*)?)(?:[Ee][-+]?\d+)?', str(toFilterObject))
    # Checks for Scientific Function  'e' in Number  --> if yes, sets the number to 0
    filteredList = [str(float(num) * 1000) if 'e' not in num.lower() else '0' for num in filteredList]
    # Adds the filtered coordinate to the String
    filteredString = ",".join(filteredList)
    # returns the finished String
    return filteredString


"""
*******************************************************************************
Function to calc average coords
*******************************************************************************
"""


def calculateAverage(points):
    return np.mean(points, axis=0)


"""
*******************************************************************************
Main Script  --- HandDetectionMP (MP-Mediapipe) 
[Detect Hands and draws Landmarks on them]
[gets coordinates of joints]
*******************************************************************************
"""


def HandDetectionMP():
    # Gets needed utils as mp_drawing
    mp_drawing = mp.solutions.drawing_utils
    # Gets solution.hands as mp_hands
    mp_hands = mp.solutions.hands
    # opens a cv2 video-capture and assigns it to cap
    cap = cv2.VideoCapture(0)
    # opens the file where the coordinates are stored
    file = open('./Data/data', 'w')
    # toggleable bool variable to choose whether to write to file or not
    doWrite = True
    # threshold value
    movement_threshold = 0.08
    # list for saving previous coordinates
    previous_hand_positions = []

    # sets up mp_hands with 1 max hand at a time and a 0.5 confidence as hands
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while cap.isOpened():
            # reads the captured frames from cap
            ret, frame = cap.read()
            if not ret:
                break

            # image blue scaling
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # processes the captured image with mp_hands
            results = hands.process(image)

            # checks if coordinates were captured
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0].landmark
                hand_positions = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                previous_hand_positions.append(hand_positions)
                max_positions = 24
                if len(previous_hand_positions) > max_positions:
                    previous_hand_positions.pop(0)
                if len(previous_hand_positions) == max_positions:
                    average_hand_position = calculateAverage(previous_hand_positions)
                    difference = np.linalg.norm(hand_positions - average_hand_position)
                    if difference < movement_threshold and doWrite:
                        print("capture")
                        data = filterNumbers(results.multi_hand_landmarks)
                        file.write(str(data))
                        file.write(" /// ")
                    # else:
                    # print("hand moving")

            # converts image back to normal color
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # draws the landmarks on the shown image for each joint
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    )
            # flips the image for easier use
            flippedImage = cv2.flip(image, 1)
            # shows the video capture with landmarks
            cv2.imshow('Raw Webcam Feed', flippedImage)
            # quits application on key "q"
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # stops video record
    cap.release()
    # closes the data file
    file.close()
    # clears all cv2 windows
    cv2.destroyAllWindows()


# Runs main script
HandDetectionMP()
