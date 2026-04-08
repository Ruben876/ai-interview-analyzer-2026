import streamlit as st
from textblob import TextBlob
import speech_recognition as sr
import cv2
import numpy as np
from deepface import DeepFace

# 🔹 Page config
st.set_page_config(page_title="AI Interview Analyzer", page_icon="🤖")

# 🎨 UI STYLE
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
    }
    .stTextArea textarea {
        background-color: #f0f2f6;
        color: black;
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 🔹 Score function
def calculate_score(sentiment, emotion):
    score = 50

    if sentiment > 0:
        score += 30
    elif sentiment < 0:
        score -= 10

    if emotion == "happy":
        score += 20
    elif emotion == "neutral":
        score += 10

    return score

# 🔹 Voice input
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = r.listen(source)

        try:
            return r.recognize_google(audio)
        except:
            return None

# 🔹 Session state
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

if "emotion" not in st.session_state:
    st.session_state.emotion = "Not detected"

# 🔹 TITLE
st.markdown("<h1 style='text-align: center;'>🤖 AI Interview Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Analyze your confidence using AI</h4>", unsafe_allow_html=True)

# 🔹 Text input
text = st.text_area("📝 Enter your answer:")

# 🔹 Voice
if st.button("🎤 Use Voice"):
    voice = get_voice_input()
    if voice:
        st.session_state.voice_text = voice
        st.success(f"You said: {voice}")
    else:
        st.error("Could not understand audio")

# 🔥 FACE + EMOTION
st.subheader("📸 Face Emotion Detection")

img_file = st.camera_input("Take a picture")

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    st.image(frame, channels="BGR")

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        st.session_state.emotion = emotion
        st.success(f"😊 Emotion detected: {emotion}")

    except:
        st.session_state.emotion = "error"
        st.error("Emotion detection failed")

# 🔹 Analyze
if st.button("Analyze 🚀"):

    input_text = text or st.session_state.voice_text or "No answer provided"

    blob = TextBlob(input_text)
    sentiment = blob.sentiment.polarity

    score = calculate_score(sentiment, st.session_state.emotion)

    st.markdown("---")

    # 💎 CARD OUTPUT
    st.markdown(f"""
    <div style="background-color:#ffffff; padding:20px; border-radius:15px; color:black;">
        <h3>📊 Final Analysis</h3>
        <p><b>Answer:</b> {input_text}</p>
        <p><b>Sentiment:</b> {sentiment}</p>
        <p><b>Emotion:</b> {st.session_state.emotion}</p>
        <p><b>Final Score:</b> {score}</p>
    </div>
    """, unsafe_allow_html=True)

    # 🎯 Feedback
    if score >= 90:
        st.success("🌟 Excellent Confidence!")
    elif score >= 70:
        st.warning("👍 Good Performance")
    else:
        st.error("⚠️ Needs Improvement")