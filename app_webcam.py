import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.set_page_config(page_title="ASL Webcam Test")

st.title("ASL Sign Language Translator")
st.write("Stage 1 - Webcam Test")

webrtc_streamer(
    key="camera-test",
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)