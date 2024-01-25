import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import os
from tkinter import Tk, filedialog, Button, Label, PhotoImage
from PIL import Image, ImageTk

model = keras.models.load_model("models/signlanguagerecognization.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)

            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            debug_image = copy.deepcopy(image)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    df = pd.DataFrame(pre_processed_landmark_list).transpose()

                    predictions = model.predict(df, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)
                    label = alphabet[predicted_classes[0]]
                    cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    print(alphabet[predicted_classes[0]])
                    print("------------------------")

            cv2.imshow('Sign Language Detector (Webcam)', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

def detect_from_images():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(title="Select Image File",
                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if not file_path:
        print("No file selected.")
        return

    image = cv2.imread(file_path)
    image = cv2.flip(image, 1)

    if image is None:
        print(f"Could not read image: {file_path}")
        return

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = copy.deepcopy(image)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                df = pd.DataFrame(pre_processed_landmark_list).transpose()

                predictions = model.predict(df, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                label = alphabet[predicted_classes[0]]
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                print(alphabet[predicted_classes[0]])
                print("------------------------")

        cv2.imshow('Sign Language Detector (Image)', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def open_file_dialog():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image File",
                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path


class App:
    def __init__(self, master):
        self.master = master
        master.title("Sign Language Detector")

        self.label = Label(master, text="Choose Detection Mode:")
        self.label.pack()

        self.webcam_button = Button(master, text="Webcam Detection", command=detect_from_webcam)
        self.webcam_button.pack()

        self.image_button = Button(master, text="Image Detection", command=self.detect_from_image)
        self.image_button.pack()

    def detect_from_image(self):
        file_path = open_file_dialog()
        if file_path:
            image = cv2.imread(file_path)
            image = cv2.flip(image, 1)

            if image is None:
                print(f"Could not read image: {file_path}")
                return

            with mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                debug_image = copy.deepcopy(image)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        df = pd.DataFrame(pre_processed_landmark_list).transpose()

                        predictions = model.predict(df, verbose=0)
                        predicted_classes = np.argmax(predictions, axis=1)
                        label = alphabet[predicted_classes[0]]
                        cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        print(alphabet[predicted_classes[0]])
                        print("------------------------")

                cv2.imshow('Sign Language Detector (Image)', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    root.geometry("300x300")
    app = App(root)
    root.mainloop()

