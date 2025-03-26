import pandas as pd
import mediapipe as mp
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pyautogui




def extract_landmarks(frame):

    #Extracts and flattens hand landmarks from a given frame
    #Returns a flattened list of the landmarks or None if no hand is detected.

    
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands_processor.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [landmark for landmark in hand_landmarks.landmark]
            landmarks_flat = [item for landmark in landmarks for item in (landmark.x, landmark.y, landmark.z)]
            return landmarks_flat
    return None





def predict_gesture(landmarks_flat):

    # Convert landmarks list to numpy array with shape (1, 63) for prediction
    #Predicts the gesture from flattened landmarks
    #Returns the predicted gesture name or None if prediction cannot be made
    
    if landmarks_flat is not None:
        
        landmarks_array = np.array(landmarks_flat).reshape(1, -1)
        prediction = model.predict(landmarks_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        gesture_name_mapping = ['thumbs_up', 'thumbs_down', 'ok', 'rock', 'point', 'v', 'fist']
        return gesture_name_mapping[predicted_class]
        
    return None



def get_bounding_box(landmarks_flat, width, height):
    if landmarks_flat:
        x_coordinates = [int(landmark_x * width) for landmark_x in landmarks_flat[0::3]]
        y_coordinates = [int(landmark_y * height) for landmark_y in landmarks_flat[1::3]]
        return (min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))
    return None, None



def start_webcam():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        frame_height, frame_width = frame.shape[:2]
        landmarks_flat = extract_landmarks(frame)
        gesture_prediction = predict_gesture(landmarks_flat)

        mirror_frame = cv2.flip(frame, 1)


        cv2.imshow('Gesture Prediction', mirror_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()