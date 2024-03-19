import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import cv2

# Function to load the YOLO model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Function to detect objects in the image
def detect_objects(model, image):
    res = model(image, stream=True)
    return res

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

    # Set confidence threshold
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.20, 0.05)

    # Live streaming camera option
    st.subheader("Live Streaming Camera")

    # WebRTC streamer to capture video from the user's camera
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=lambda: VideoTransformer(model, conf_threshold),
        async_transform=True,
    )

    if webrtc_ctx.video_processor:
        st.video(webrtc_ctx.video_processor.frame_out)

class VideoTransformer:
    def __init__(self, model, conf_threshold):
        self.model = model
        self.conf_threshold = conf_threshold

    def transform(self, frame):
        # Convert frame to BGR format
        image = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_RGB2BGR)
        
        # Detect objects in the frame
        results = detect_objects(self.model, image)
        
        # Draw bounding boxes and labels
        for r in results:
            image = r.render()

        # Convert back to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Return the transformed frame
        return image

if __name__ == "__main__":
    main()
