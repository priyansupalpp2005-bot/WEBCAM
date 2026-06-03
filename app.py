#rebuild
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp

st.set_page_config(page_title="ASL Hand Detection")

st.title("ASL Sign Language Translator")
st.write("Stage 2 - Hand Detection")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class HandDetector(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="hand-detection",
    video_processor_factory=HandDetector,
    media_stream_constraints={"video": True, "audio": False},
)