import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Use a face detection cascade for testing purposes
# Replace with your insect cascade XML file once it's available
insect_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # For testing purposes

# Check if cascade loaded successfully
if insect_cascade.empty():
    print("Error: Unable to load the insect cascade classifier.")
    exit()

# Function to detect insect in an image
def detect_insect(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect insects using the Haar Cascade classifier
    insects = insect_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(insects) > 0:
        for (x, y, w, h) in insects:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around detected insect
        cv2.putText(image, "Insect Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Insect detected!")
    else:
        cv2.putText(image, "No Insect Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("No insect detected.")
    
    # Show the image with or without insect detected
    cv2.imshow('Insect Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to open a file dialog to select the image
def upload_image():
    # Set up Tkinter root window (not displayed)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # Open file dialog to let user select an image
    file_path = filedialog.askopenfilename(title="Select Insect Image", filetypes=(("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")))
    if file_path:
        detect_insect(file_path)

# Run the insect detection by uploading an image
upload_image()
