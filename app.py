import streamlit as st
import joblib

st.set_page_config(page_title="ASL Translator")

st.title("ASL Sign Language Translator")

model = joblib.load("gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.success("Model loaded successfully!")

st.write("Classes:")
st.write(label_encoder.classes_)