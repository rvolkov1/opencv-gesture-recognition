import cv2
import mediapipe as mp
from data import Data
from model import GestureModel
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands  = mp.solutions.hands

cap = cv2.VideoCapture(0)

data = Data()
gesture_model = GestureModel()


with mp_hands.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5,
    max_num_hands = 1
) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("ignoring empty camera frame")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        k = cv2.waitKey(33)

        if(k == 107):
            data.toggle_collection()

        # draw hand annotations on image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            if (k >= 49 and k <= 53):
                # 1 is 49
                data.add_data_point(k-49, data.landmarks_to_list(results.multi_hand_landmarks))

            prediction = gesture_model.model.predict(np.array([np.array(data.landmarks_to_list(results.multi_hand_landmarks))]))

            print("prediction", gesture_model.gesture_names[np.argmax(prediction)])

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))        

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()