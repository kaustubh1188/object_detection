import streamlit as st
import cv2
import math
from PIL import Image
import io
import requests
import os
from numpy import random
from ultralytics import YOLO

# Set environment variable to suppress OpenCV warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to predict objects in the image
def predict_image(model, image, conf_threshold, iou_threshold):
    # Predict objects using the model
    res = model.predict(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        device='cpu',
    )

    class_name = model.model.names
    classes = res[0].boxes.cls
    class_counts = {}

    # Count the number of occurrences for each class
    for c in classes:
        c = int(c)
        class_counts[class_name[c]] = class_counts.get(class_name[c], 0) + 1

    # Generate prediction text
    prediction_text = 'Predicted '
    for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
        prediction_text += f'{v} {k}'

        if v > 1:
            prediction_text += 's'

        prediction_text += ', '

    prediction_text = prediction_text[:-2]
    if len(class_counts) == 0:
        prediction_text = "No objects detected"

    # Calculate inference latency
    latency = sum(res[0].speed.values())  # in ms, need to convert to seconds
    latency = round(latency / 1000, 2)
    prediction_text += f' in {latency} seconds.'

    # Convert the result image to RGB
    res_image = res[0].plot()
    res_image = cv2.cvtColor(res_image, cv2.COLOR_BGR2RGB)

    return res_image, prediction_text

# Function for live streaming detection
def live_streaming_detection(model, conf_threshold, iou_threshold, video_url):
    # start streaming from the camera
    cap = cv2.VideoCapture(video_url)

    # Check if the video stream is opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open video stream.")
        return

    # Object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush","fire","smoke","gun","animal"
                  ]

    # Placeholder to display the video stream
    placeholder = st.empty()

    # Loop to continuously read frames from the camera and update the Streamlit app
    while True:
        # Read a frame from the camera
        ret, img = cap.read()

        # Check if the frame was read successfully
        if not ret:
            st.error("Error: Failed to read frame from video stream.")
            break

        # Get predictions from the model
        results = model(img, stream=True)

        # Draw bounding boxes and labels
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # Object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        # Display the frame with bounding boxes in Streamlit
        placeholder.image(img, channels="BGR", use_column_width=True)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Set Streamlit page configuration
    st.set_page_config(
        page_title="Wildfire Detection",
        page_icon="🔥",
    )

    # App title
    st.title("Wildfire Detection")

    # Model selection
    models_dir = "yolo-Weights"
    selected_model = "yolov8n"
    model_path = os.path.join(models_dir, selected_model + ".pt")
    model = load_model(model_path)

    # Set confidence and IOU thresholds
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.5, 0.05)

    # Live streaming camera option
    st.subheader("Live Streaming Camera")
    video_url = st.text_input("Enter the IP or URL of the camera:")
    if st.button("Start Live Stream"):
        live_streaming_detection(model, conf_threshold, iou_threshold, video_url)

if __name__ == "__main__":
    main()
