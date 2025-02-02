import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Model parameters
img_height, img_width = 150, 150

# Load the trained model
model_path = 'car_accident_detector_model.h5'
model = tf.keras.models.load_model(model_path)

# Function to predict an image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    
    if prediction[0][0] > 0.5:
        return f"Non-Accident (Confidence: {prediction[0][0]*100:.2f}%)"
    else:
        return f"Accident (Confidence: {(1-prediction[0][0])*100:.2f}%)"

# Function to open file dialog and display prediction
def open_file_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        # Display the image
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        
        # Predict and display the result
        result = predict_image(file_path)
        result_label.config(text=f"Prediction: {result}")

# Create the GUI
root = tk.Tk()
root.title("Car Accident Detection")

# Add GUI components
frame = tk.Frame(root)
frame.pack(pady=20)

btn = tk.Button(frame, text="Select an Image", command=open_file_and_predict)
btn.pack()

img_label = tk.Label(frame)
img_label.pack(pady=10)

result_label = tk.Label(frame, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

# Run the GUI loop
root.mainloop()
