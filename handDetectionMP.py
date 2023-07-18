import mediapipe as mp
import cv2
import re

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
file = open('./Data/data', 'w')
doWrite = True
compareList = 0
tolerance = 0.01
results = 0


def filterNumbers(toFilterObject):
    filteredList = re.findall(r'([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?', str(toFilterObject))
    filteredString = ""
    for i in (str(filteredList)):
        if i != '[' and i != '(' and i != ')' and i != ']' and i != '\'' and i != ' ':
            if i == ',':
                



            filteredString = filteredString + str(i)
    return filteredString


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        compareList = results
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if doWrite:
            if results.multi_hand_landmarks:
                data = filterNumbers(results.multi_hand_landmarks)
                file.write(str(data))

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

cap.release()
file.close()
cv2.destroyAllWindows()
