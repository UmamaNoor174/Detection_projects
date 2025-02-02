import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model (make sure to provide the correct path to your model file)
model = load_model('bone_fracture_detection_model.h5')

# Function to predict fracture in the uploaded image
def predict_image(img_path):
    # Load the image and preprocess it
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Return prediction result
    if prediction[0][0] > 0.5:
        return "No Fracture Detected!", img_path
    else:
        return "Fracture Detected!", img_path

# Function to open file dialog, display the image, and predict
def open_image():
    # Ask the user to select an image file
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        # Get prediction result
        result, img_path = predict_image(file_path)
        
        # Display the image
        display_image(img_path)
        
        # Show prediction result
        result_label.config(text=f"Prediction: {result}")

# Function to display the uploaded image
def display_image(img_path):
    # Open and resize the image
    img = Image.open(img_path)
    img.thumbnail((300, 300))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    
    # Update the label to show the image
    image_label.config(image=img_tk)
    image_label.image = img_tk

# Create the main window
root = tk.Tk()
root.title("Bone Fracture Detection")
root.geometry("400x500")

# Add a label to the window
title_label = tk.Label(root, text="Bone Fracture Detection", font=("Arial", 16))
title_label.pack(pady=10)

# Add a label for displaying the image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Add a label for showing the prediction result
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

# Add a button to open the file dialog
button = tk.Button(root, text="Upload Image", font=("Arial", 14), command=open_image)
button.pack(pady=20)

# Start the GUI event loop
root.mainloop()
