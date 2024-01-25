import tkinter as tk
from tkinter import filedialog, Toplevel
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from threading import Thread

class ObjectDetection:
    def __init__(self, root):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.root = root
        self.root.title("YOLOv5 Object Detection")

        self.button_image = tk.Button(root, text="Detect from Image", command=self.detect_image)
        self.button_image.pack(pady=5)

        self.button_video = tk.Button(root, text="Detect from Video", command=self.detect_video)
        self.button_video.pack(pady=5)

        self.button_webcam = tk.Button(root, text="Detect from Webcam", command=self.detect_webcam)
        self.button_webcam.pack(pady=5)

        self.is_detecting_video = False
        self.video_thread = None

        self.video_toplevel = None  
        self.video_label = None     

        self.video_after_id = None  

    def load_model(self):
        return torch.hub.load('ultralytics/yolov5', 'custom', 'models/best.pt')

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def detect_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            image = cv2.imread(file_path)
            results = self.score_frame(image)
            image_with_boxes = self.plot_boxes(results, image)
            self.show_image_in_separate_window(image_with_boxes)

    def detect_video(self):
        file_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        if file_path:
            self.is_detecting_video = True
            self.video_toplevel = Toplevel(self.root)
            self.video_toplevel.title("YOLOv5 Object Detection Video Result")
            self.video_label = tk.Label(self.video_toplevel)
            self.video_label.pack()
            self.video_thread = Thread(target=self.detect_video_thread, args=(file_path,))
            self.video_thread.start()

    def detect_video_thread(self, video_path):
        cap = cv2.VideoCapture(video_path)

        def update_video_frame():
            ret, frame = cap.read()
            if ret:
                results = self.score_frame(frame)
                frame_with_boxes = self.plot_boxes(results, frame)
                self.show_video_frame(frame_with_boxes)
                self.video_after_id = self.root.after(60, update_video_frame)
            else:
                self.is_detecting_video = False
                cap.release()
                self.video_toplevel.destroy()

        update_video_frame()

    def detect_webcam(self):
        self.is_detecting_video = True
        self.video_toplevel = Toplevel(self.root)
        self.video_toplevel.title("YOLOv5 Object Detection Webcam Result")
        self.video_label = tk.Label(self.video_toplevel)
        self.video_label.pack()
        self.video_thread = Thread(target=self.detect_webcam_thread)
        self.video_thread.start()

    def detect_webcam_thread(self):
        cap = cv2.VideoCapture(0)  

        def update_video_frame():
            nonlocal cap  

            ret, frame = cap.read()
            if ret:
                results = self.score_frame(frame)
                frame_with_boxes = self.plot_boxes(results, frame)
                self.show_video_frame(frame_with_boxes)
                self.video_after_id = self.root.after(60, update_video_frame)
            elif self.is_detecting_video:  
                update_video_frame()  
            else:
                cap.release()
                self.video_toplevel.destroy()

        update_video_frame()  
        
          

    def show_video_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        photo_image = ImageTk.PhotoImage(image=img)

        self.video_label.config(image=photo_image)
        self.video_label.image = photo_image

    def show_image_in_separate_window(self, frame):
        window = Toplevel(self.root)
        window.title("YOLOv5 Object Detection Result")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        photo_image = ImageTk.PhotoImage(image=img)

        label = tk.Label(window, image=photo_image)
        label.photo_image = photo_image
        label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("300x300")
    detection_app = ObjectDetection(root)
    root.mainloop()



