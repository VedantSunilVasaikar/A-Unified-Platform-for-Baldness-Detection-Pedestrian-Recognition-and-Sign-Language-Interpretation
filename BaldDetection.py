import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/best.pt")

def load_and_preprocess_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (192, 192))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)

        image_label.config(image=photo)
        image_label.image = photo
        image_label.file_path = file_path
        detect_baldness()

def detect_baldness():
    file_path = image_label.file_path
    frame = cv2.imread(file_path)

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    results = model.predict(frame_pil)

    for r in results:
        im_array = r.plot() 
        detected_image = Image.fromarray(im_array[..., ::-1])  

        detected_image.thumbnail((300, 300))

        detected_photo = ImageTk.PhotoImage(detected_image)

        image_label.config(image=detected_photo)
        image_label.image = detected_photo

root = tk.Tk()
root.title("Baldness Detection")
root.geometry("720x720")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()






