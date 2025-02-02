import cv2
import tensorflow as tf
import numpy as np
from tkinter import Tk, Button, filedialog, Label
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model(r"C:\Users\Admin\Desktop\projec-AI\myenv\venv\leaf_plan_detection_model.h5")
print(model.summary())
# Load the trained model
print("Model loaded successfully!")

# Updated Class Names
class_names = [
    'Chinar healthy (P11a)',
    'Gauva healthy (P3a)',
    'Jamun healthy (P5a)',
    'Mango healthy (P0a)'
]

# Function: Real-time Webcam Detection with Green Detection
def webcam_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV to detect green regions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])  # Lower bound for green in HSV
        upper_green = np.array([85, 255, 255])  # Upper bound for green in HSV
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours of the green regions
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If a significant green area is found, make predictions
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 1000:  # Trigger threshold for green region
                # Preprocess the frame
                resized_frame = cv2.resize(frame, (150, 150))  # Match model input size
                img_array = np.expand_dims(resized_frame, axis=0)
                img_array = img_array / 255.0

                # Predict class
                predictions = model.predict(img_array)
                confidence = np.max(predictions)
                if confidence > 0.6:  # Confidence threshold
                    predicted_class = class_names[np.argmax(predictions)]
                else:
                    predicted_class = "Uncertain"

                # Display prediction on frame
                cv2.putText(frame, f"Prediction: {predicted_class} ({confidence*100:.2f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the webcam frame
        cv2.imshow("Real-time Webcam Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Function: Image Recognition
def image_recognition():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        print("No file selected.")
        return

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Match preprocessing

    # Predict class
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    if confidence > 0.6:  # Confidence threshold
        predicted_class = class_names[np.argmax(predictions)]
    else:
        predicted_class = "Uncertain"

    # Debugging: Print all class confidences
    print("Class Confidences:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {predictions[0][i]*100:.2f}%")

    # Display result
    print(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
    result_label.config(text=f"Prediction: {predicted_class} ({confidence*100:.2f}%)", fg="green")

    # Display the uploaded image
    img_display = Image.open(file_path)
    img_display.thumbnail((300, 300))  # Resize to fit in GUI
    img_tk = ImageTk.PhotoImage(img_display)
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Main GUI Function
def main_gui():
    global result_label, img_label
    # Create a GUI window
    root = Tk()
    root.title("Plant Leaf Detection")

    # Add Label
    label = Label(root, text="Choose Detection Mode", font=("Helvetica", 16))
    label.pack(pady=10)

    # Real-time Webcam Detection Button
    webcam_button = Button(root, text="Real-time Webcam Detection", font=("Helvetica", 12),
                           command=webcam_detection)
    webcam_button.pack(pady=10)

    # Image Recognition Button
    image_button = Button(root, text="Image Recognition", font=("Helvetica", 12),
                          command=image_recognition)
    image_button.pack(pady=10)

    # Label to Display Results
    result_label = Label(root, text="", font=("Helvetica", 14))
    result_label.pack(pady=10)

    # Label to Display the Image
    img_label = Label(root)
    img_label.pack(pady=10)

    # Start the GUI loop
    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    main_gui()
