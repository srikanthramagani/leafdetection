import streamlit as st
import google.generativeai as genai
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
from googletrans import Translator
from gtts import gTTS
import speech_recognition as sr
import os
import tempfile
import time
import webbrowser
import re

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAVqlQbXpkkHgC-Wi6WnSrdp0-jzzysDRE"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# Translator Setup
translator = Translator()
languages = {'English': 'en', 'Hindi': 'hi', 'Telugu': 'te', 'Tamil': 'ta', 'Kannada': 'kn'}

# Load YOLO model once
def load_model():
    if "model" not in st.session_state:
        st.session_state.model = YOLO(r"leaf.pt")
    return st.session_state.model

# Function to fetch disease solution
def get_disease_solution(disease_name, lang_code):
    prompt = f"Describe the symptoms, natural remedies, chemical treatments, and prevention measures for {disease_name} in simple farmer-friendly language."
    try:
        response = gemini_model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            translated_text = translator.translate(response.text, dest=lang_code).text
            return translated_text
    except:
        return "No treatment details available at the moment."
    return "No treatment details available."

# Function to extract chemicals from remedies
def extract_chemical_treatment(solution):
    chemical_keywords = ["pesticide", "herbicide", "fungicide", "chemical", "spray", "treatment"]
    words = solution.split()
    for i, word in enumerate(words):
        if any(kw in word.lower() for kw in chemical_keywords) and i + 1 < len(words):
            return words[i + 1]  # Assume next word is the chemical name
    return None

# Function to convert text to speech
def speak_text(text, lang_code):
    try:
        tts = gTTS(text=text.replace("*", "").replace("#", "").replace("_", "").replace("\n", ". "), lang=lang_code)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format='audio/mp3')
        time.sleep(2)
        os.remove(temp_audio.name)
    except:
        st.error("Error in text-to-speech.")

# Function to fetch Amazon search link
def get_amazon_link(chemical_name):
    return f"https://www.amazon.com/s?k={chemical_name.replace(' ', '+')}"

# Function to fetch a treatment video
def get_treatment_video_link(disease_name):
    return f"https://www.google.com/search?q={disease_name}+treatment+video"

# Streamlit UI
st.set_page_config(page_title="Farmer Friend", layout="wide")
st.title("🚜 Farmer Friend - Leaf Disease Detection")

# Sidebar Options
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📷 Upload Image", "🎥 Live Capture", "🎤 Voice Input"])
selected_language = st.sidebar.selectbox("🌍 Select Language", list(languages.keys()))
lang_code = languages[selected_language]

# 📷 Upload Image Mode
if menu == "📷 Upload Image":
    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        model = load_model()
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
        results = model(image)
        if results[0].boxes:
            disease_name = results[0].names[int(results[0].boxes[0].cls)]
            solution = get_disease_solution(disease_name, lang_code)
            st.subheader(f"🌿 Disease: {disease_name}")
            st.write(solution)
            if st.button(f"🔊 Hear Remedy for {disease_name}"):
                speak_text(solution, lang_code)

            # Extract chemical and show cart button
            chemical_name = extract_chemical_treatment(solution)
            if chemical_name:
                st.markdown(f"**🛒 Recommended Chemical:** {chemical_name}")
                if st.button(f"🛒 Buy {chemical_name} on Amazon"):
                    webbrowser.open(get_amazon_link(chemical_name))

            if st.button("📺 Watch Treatment Video"):
                webbrowser.open(get_treatment_video_link(disease_name))

# 🎥 Live Capture Mode
if menu == "🎥 Live Capture":
    if st.button("📷 Capture & Analyze"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            model = load_model()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.image(image, caption="📷 Captured Image", use_container_width=True)
            results = model(image)
            if results[0].boxes:
                disease_name = results[0].names[int(results[0].boxes[0].cls)]
                solution = get_disease_solution(disease_name, lang_code)
                st.subheader(f"🌿 Disease: {disease_name}")
                st.write(solution)
                if st.button(f"🔊 Hear Remedy for {disease_name}"):
                    speak_text(solution, lang_code)

                # Extract chemical and show cart button
                chemical_name = extract_chemical_treatment(solution)
                if chemical_name:
                    st.markdown(f"**🛒 Recommended Chemical:** {chemical_name}")
                    if st.button(f"🛒 Buy {chemical_name} on Amazon"):
                        webbrowser.open(get_amazon_link(chemical_name))

                if st.button("📺 Watch Treatment Video"):
                    webbrowser.open(get_treatment_video_link(disease_name))

# 🎤 Voice Input Mode
if menu == "🎤 Voice Input":
    if st.button("🎙 Speak & Get Remedies"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                disease_name = recognizer.recognize_google(audio)
                st.write(f"📝 You said: **{disease_name}**")
                solution = get_disease_solution(disease_name, lang_code)
                st.subheader(f"🌿 Disease: {disease_name}")
                st.write(solution)
                if st.button(f"🔊 Hear Remedy for {disease_name}"):
                    speak_text(solution, lang_code)

                # Extract chemical and show cart button
                chemical_name = extract_chemical_treatment(solution)
                if chemical_name:
                    st.markdown(f"**🛒 Recommended Chemical:** {chemical_name}")
                    if st.button(f"🛒 Buy {chemical_name} on Amazon"):
                        webbrowser.open(get_amazon_link(chemical_name))

                if st.button("📺 Watch Treatment Video"):
                    webbrowser.open(get_treatment_video_link(disease_name))
            except:
                st.error("Could not recognize speech. Try again.")
