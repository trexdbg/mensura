import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import av
import threading
import os
from typing import Union

st.title("OpenCV Filters on Video Stream")

class VideoTransformer(VideoTransformerBase):

    def __init__(self):
        self.frame_lock = threading.Lock()
        self.out_image = None
        self.filter = "none"  # Ajout de l'attribut filter à la classe

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.filter == "blur":
            img = cv2.GaussianBlur(img, (21, 21), 0)
        elif self.filter == "canny":
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        elif self.filter == "grayscale":
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif self.filter == "sepia":
            kernel = np.array(
                [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
            )
            img = cv2.transform(img, kernel)
        elif self.filter == "invert":
            img = cv2.bitwise_not(img)
        
        with self.frame_lock:
            self.out_image = img
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Interface Streamlit
ctx = webrtc_streamer(
    key="streamer",
    video_processor_factory=VideoTransformer,
    sendback_audio=False
)

if ctx and ctx.video_transformer:
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])

    with col1:
        if st.button("None"):
            ctx.video_transformer.filter = "none"
    with col2:
        if st.button("Blur"):
            ctx.video_transformer.filter = "blur"
    with col3:
        if st.button("Grayscale"):
            ctx.video_transformer.filter = "grayscale"
    with col4:
        if st.button("Sepia"):
            ctx.video_transformer.filter = "sepia"
    with col5:
        if st.button("Canny"):
            ctx.video_transformer.filter = "canny"
    with col6:
        if st.button("Invert"):
            ctx.video_transformer.filter = "invert"

    snap = st.button("Capture")
    if snap:
        with ctx.video_transformer.frame_lock:
            out_image = ctx.video_transformer.out_image
        
        if out_image is not None:
            st.image(out_image, channels="BGR")
            # save_path = os.path.join(os.getcwd(), "captured_image.jpg")
            # cv2.imwrite(save_path, out_image)
            # st.success(f"Image saved at {save_path}")
        else:
            st.warning("No frames available yet.")
