import mediapipe as mp
import cv2
import re

file = open('./Data/dataSTATIC', 'w')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
img = cv2.imread('./StaticSamples/A.jpg')


def filterNumbers(toFilterObject):
    filteredList = re.findall(r'[-+]?(?:\.\d+|\d+(?:\.\d*)?)(?:[Ee][-+]?\d+)?', str(toFilterObject))
    filteredList = [str(float(num) * 1000) if 'e' not in num.lower() else '0' for num in filteredList]
    filteredString = ",".join(filteredList)
    return filteredString


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    data = filterNumbers(results.multi_hand_landmarks)
    file.write(str(data))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            )
    while True:
        cv2.imshow('Raw Sample Image', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
