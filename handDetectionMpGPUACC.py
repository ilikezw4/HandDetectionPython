import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Check if CUDA is available
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("CUDA GPU acceleration is not available.")
    exit()

# Transfer image to GPU memory
gpu_frame = cv2.cuda_GpuMat()

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Transfer frame to GPU memory
        gpu_frame.upload(frame)

        # Convert image to RGB and process on GPU
        gpu_rgb_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(gpu_rgb_frame)

        # Download processed image from GPU memory
        cpu_rgb_frame = gpu_rgb_frame.download()

        # Convert image to BGR for rendering
        image = cv2.cvtColor(cpu_rgb_frame, cv2.COLOR_RGB2BGR)

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
cv2.destroyAllWindows()
