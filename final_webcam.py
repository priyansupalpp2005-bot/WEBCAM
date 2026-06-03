import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
import threading
from collections import deque, Counter
from datetime import datetime

# =========================
# LOAD MODEL
# =========================
model = joblib.load("gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================
# SPEECH ENGINE
# =========================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# CAMERA
cap = cv2.VideoCapture(0)

cv2.namedWindow("ASL INDUSTRY SYSTEM", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL INDUSTRY SYSTEM", 640, 480)

# =========================
# VARIABLES
# =========================
sentence = ""
prediction_history = deque(maxlen=25)

last_added = ""
last_time = 0
last_seen = time.time()

COOLDOWN = 2.5
INACTIVITY_GAP = 2.0
CONF_THRESHOLD = 0.95
AUTO_SPEECH = True

WORD_LIST = [
    "hello", "thanks", "please", "yes", "no", "love", "help",
    "sorry", "good", "morning", "night", "family", "friend",
    "computer", "language", "text", "sign", "speech", "learn",
    "you", "me", "we", "word", "read", "write", "language",
    "computer", "technology", "music", "happy", "birthday",
    "coffee", "water", "food", "study", "school", "teacher",
    "student", "work", "home", "house", "travel", "movie",
    "city", "country", "market", "money", "help", "stop",
    "start", "open", "close", "learn", "talk", "listen",
    "watch", "think", "friendship", "family", "future",
    "today", "tomorrow", "yesterday", "answer", "question"
]

# =========================
# SAVE HISTORY
# =========================
def save_history(text):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.now()} - {text}\n")

# =========================
# SPEAK
# =========================
def speak(text):
    engine.say(text)
    engine.runAndWait()


def speak_text(text):
    if text:
        threading.Thread(target=speak, args=(text,), daemon=True).start()

# =========================
# STABLE PREDICTION
# =========================
def get_stable():
    if len(prediction_history) == 0:
        return None
    return Counter(prediction_history).most_common(1)[0][0]

# =========================
# WORD SUGGESTION
# simple AI-like completion for current partial word

def get_suggestions(text):
    fragment = text.strip().split()[-1] if text and not text.endswith(" ") else ""
    if not fragment:
        return []
    fragment = fragment.lower()
    suggestions = sorted([w for w in WORD_LIST if w.startswith(fragment)])
    return suggestions[:3]

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = None
    confidence = 0

    # =========================
    # HAND DETECTION + DRAW FIX
    # =========================
    if result.multi_hand_landmarks:
        last_seen = time.time()

        for hand_landmarks in result.multi_hand_landmarks:

            # ⭐ THIS FIXES YOUR ISSUE (HAND VISIBILITY)
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            landmarks = np.array(landmarks)
            landmarks = landmarks - landmarks[0]
            landmarks = landmarks.reshape(1, -1)

            prediction = model.predict(landmarks)[0]

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(landmarks)[0]
                confidence = np.max(probs)
            else:
                confidence = 1.0

            prediction = label_encoder.inverse_transform([prediction])[0]
            prediction_history.append(prediction)

    stable = get_stable()
    suggestions = get_suggestions(sentence)
    suggestion_text = ", ".join(suggestions) if suggestions else "No suggestion"
    current_time = time.time()

    # =========================
    # WORD SPACE DETECTION
    # =========================
    if current_time - last_seen > INACTIVITY_GAP:
        if sentence and not sentence.endswith(" "):
            sentence += " "

    # =========================
    # LETTER ADDING LOGIC
    # =========================
    if stable and confidence > CONF_THRESHOLD and current_time - last_time > COOLDOWN:

        if stable != last_added:
            if stable == "nothing":
                pass
            elif stable == "del":
                if sentence:
                    sentence = sentence[:-1]
                last_added = stable
                last_time = current_time
            elif stable == "space":
                if not sentence.endswith(" "):
                    sentence += " "
                last_added = stable
                last_time = current_time
                if AUTO_SPEECH:
                    last_word = sentence.strip().split()[-1] if sentence.strip() else ""
                    if last_word:
                        speak_text(last_word)
            else:
                sentence += stable
                last_added = stable
                last_time = current_time

    # =========================
    # UI
    # =========================
    cv2.rectangle(frame, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.putText(frame, "Sign Language to Text Converter", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Character: {prediction if prediction else '-'}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (330, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Stable: {stable if stable else '-'}", (520, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Press S to speak sentence", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {sentence[-60:]}", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"AI Suggestions: {suggestion_text}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("ASL INDUSTRY SYSTEM", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    elif key == ord('c'):
        sentence = ""

    elif key == ord('s'):
        if sentence.strip():
            speak_text(sentence)
            save_history(sentence)

cap.release()
cv2.destroyAllWindows()