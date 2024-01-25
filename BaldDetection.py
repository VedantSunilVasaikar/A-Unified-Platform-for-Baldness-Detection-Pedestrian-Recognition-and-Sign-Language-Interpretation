import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model('models/baldnessdetection.h5')

def load_and_preprocess_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_baldness(image):
    prediction = model.predict(image)
    return prediction[0][0]

def upload_image():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    image.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo)
    image_label.image = photo
    image_label.file_path = file_path
    detect_button.config(state=tk.NORMAL)

def detect_baldness():
    file_path = image_label.file_path
    image = load_and_preprocess_image(file_path)
    prediction = predict_baldness(image)

    if prediction > 0.5:
        result_label.config(text="Bald")
    else:
        result_label.config(text="Not Bald")

root = tk.Tk()
root.title("Baldness Detection")
root.geometry("720x720")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

detect_button = tk.Button(root, text="Detect Baldness", command=detect_baldness, state=tk.DISABLED)
detect_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()
