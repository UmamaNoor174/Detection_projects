import cv2
import torch
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
import numpy as np

# Function to detect objects and colors
def detect_objects_and_colors(image_path):
    # Load YOLOv5 pre-trained model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pre-trained YOLOv5
    
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = model(image_rgb)  # Detect objects
    
    # Get results: bounding boxes, labels, confidence
    detections = results.pandas().xyxy[0]
    
    for index, row in detections.iterrows():
        x1, y1, x2, y2, confidence, class_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
        
        # Extract the detected region
        roi = image_rgb[y1:y2, x1:x2]
        
        # Find the dominant color
        dominant_color = get_dominant_color(roi)
        
        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} ({dominant_color})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display the image in a new window
    cv2.imshow("Object and Color Detection", image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# Function to get the dominant color in an image region
def get_dominant_color(image):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Use KMeans clustering to find the dominant color
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[0]
    
    # Convert to integer RGB values
    return tuple(map(int, dominant_color))

# Main function to open a file dialog and detect objects
if __name__ == "__main__":
    # Create a simple Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open a file dialog to select an image
    print("Select an image file for detection:")
    image_path = filedialog.askopenfilename(title="Select an Image",
                                            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])

    if image_path:
        try:
            detect_objects_and_colors(image_path)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No file selected!")
