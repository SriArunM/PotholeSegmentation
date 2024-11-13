import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
from pathlib import Path
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration

path = "C:/Users/msria/Downloads/Potholefinal2.pt"


# Define the video processor class for real-time webcam
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, batch_size=1, conf_threshold=0.25, frame_skip=1):
        # Initialize the YOLO model without weights_only (since it's not supported)
        self.model = YOLO(path)
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.frame_count = 0  # To track frames for frame skipping

    def recv(self, frame):
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Perform object detection with the specified confidence threshold and batch size
        results = self.model(img, conf=self.conf_threshold, batch=self.batch_size)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def process_video(uploaded_video, conf_threshold, batch_size, model):
    # Temporary file to store the input and output video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as input_file:
        input_file.write(uploaded_video.read())
        temp_video_path = input_file.name

    # Open the input video using OpenCV
    cap = cv2.VideoCapture(temp_video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Temporary file to store the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_file:
        output_video_path = output_file.name

    # Open video writer to save the output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    st_frame = st.empty()

    # Read and process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection with the specified confidence threshold and batch size
        results = model(frame, conf=conf_threshold, batch=batch_size)

        # Annotate the frame with bounding boxes
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Convert the frame back to RGB for display in Streamlit
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit app
        st_frame.image(annotated_frame_rgb, channels="RGB")

    cap.release()
    out.release()

    # Return the output video path for download
    return output_video_path


def main():
    st.title("YOLO Object Detection with Streamlit")

    # Sidebar for user selection
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Select input type:", ("Real-time Camera", "Upload Image", "Upload Video")
    )

    # Batch size and confidence threshold input
    batch_size = st.sidebar.slider(
        "Batch Size", min_value=1, max_value=32, value=1, step=1
    )
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05
    )

    # Frame skipping option (optional)
    frame_skip = st.sidebar.number_input(
        "Frame Skip (Set to 1 for no skipping)", min_value=1, max_value=10, value=1
    )

    # Initialize YOLO model without weights_only
    model = YOLO(path)

    if option == "Real-time Camera":
        st.write("Real-time object detection using webcam")
        # Use WebRTC for real-time video streaming
        webrtc_streamer(
            key="yolo",
            video_processor_factory=lambda: YOLOVideoProcessor(
                batch_size=batch_size,
                conf_threshold=conf_threshold,
                frame_skip=frame_skip,
            ),
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
        )

    elif option == "Upload Image":
        st.write("Upload an image for object detection")
        uploaded_image = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            # Convert the uploaded file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Perform object detection with the specified confidence threshold and batch size
            results = model(image, conf=conf_threshold, batch=batch_size)
            annotated_image = results[0].plot()

            # Display the image
            st.image(annotated_image, channels="BGR", caption="Processed Image")

    elif option == "Upload Video":
        st.write("Upload a video for object detection")
        uploaded_video = st.file_uploader(
            "Choose a video...", type=["mp4", "mov", "avi"]
        )

        if uploaded_video is not None:
            st.write("Processing video...")
            output_video_path = process_video(
                uploaded_video, conf_threshold, batch_size, model
            )

            # Provide the download link for the processed video
            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="Download Processed Video",
                    data=f,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                )


if __name__ == "__main__":
    main()
