import tkinter as tk
from tkinter import ttk
import os
import subprocess
import sys

class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Main Window")

        
        self.subprocesses = []

        
        self.style = ttk.Style()
        self.style.configure("TButton",
                             font=("Arial", 14),
                             padding=10,
                             width=30)

        self.setup_ui()

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        instructions_label = ttk.Label(self.master, text="Select a Program:", font=("Arial", 16))
        instructions_label.grid(row=0, column=0, pady=20, columnspan=2)

        button_frame = ttk.Frame(self.master)
        button_frame.grid(row=1, column=0, pady=10, padx=10)

        Baldness_button = ttk.Button(button_frame, text="Baldness Detection", command=self.launch_BaldnessDetection, style="TButton")
        Baldness_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        Pedestrian_button = ttk.Button(button_frame, text="Pedestrian Detection", command=self.launch_PedestrianDetection, style="TButton")
        Pedestrian_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        SignLanguage_button = ttk.Button(button_frame, text="Sign language Recognization", command=self.launch_SignLanguage, style="TButton")
        SignLanguage_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
  
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

    def launch_BaldnessDetection(self):       
        subprocess_instance = subprocess.Popen(["python", "BaldDetection.py"])
        self.subprocesses.append(subprocess_instance)

    def launch_PedestrianDetection(self):
        subprocess_instance = subprocess.Popen(["python", "pedestrains.py"])
        self.subprocesses.append(subprocess_instance)

    def launch_SignLanguage(self):       
        subprocess_instance = subprocess.Popen(["python", "SignLanguageRecognization.py"])
        self.subprocesses.append(subprocess_instance)

    

    def on_close(self):
        for subprocess_instance in self.subprocesses:
            subprocess_instance.kill()

        self.master.destroy()
        sys.exit()

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()