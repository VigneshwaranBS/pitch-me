import os

# Set the path to the ffmpeg executable
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"  # Replace with the actual path to ffmpeg

from flask import Flask, render_template, request, redirect, url_for
import cv2
import mediapipe as mp
import tensorflow_hub as hub
import tensorflow as tf
import time
import numpy as np
import moviepy.editor as mp_editor
import speech_recognition as sr


# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv"}

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# MediaPipe and YAMNet setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)
model = hub.load("https://tfhub.dev/google/yamnet/1")

action_log = []
speech_log_file = os.path.join(app.config["UPLOAD_FOLDER"], "speech_log.txt")

def allowed_file(filename):
    """Check if the file has a valid extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def preprocess_audio(audio_data):
    """Preprocess audio data to fit YAMNet input."""
    audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    return tf.reshape(audio_tensor, [-1])

def log_action(action, start_time):
    """Log detected actions with a timestamp."""
    timestamp = time.time() - start_time
    action_log.append((timestamp, action))
    print(f"At {timestamp:.2f} seconds: {action}")

def detect_actions(results, start_time):
    """Detect actions based on MediaPipe results."""
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        nose = pose_landmarks[mp_holistic.PoseLandmark.NOSE]
        if nose.y < 0.5:
            log_action("Head above midpoint", start_time)

    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        left_eye_ratio = face_landmarks[159].y - face_landmarks[145].y
        right_eye_ratio = face_landmarks[386].y - face_landmarks[374].y
        if left_eye_ratio < 0.02:
            log_action("Left eye blinked", start_time)
        if right_eye_ratio < 0.02:
            log_action("Right eye blinked", start_time)

def save_logs():
    """Save action and speech logs."""
    with open(os.path.join(app.config["UPLOAD_FOLDER"], "action_log.txt"), "w") as f:
        for timestamp, action in action_log:
            f.write(f"{timestamp:.2f}: {action}\n")
    print("Action log saved.")

def convert_speech_to_text(audio_file):
    """Convert audio data to text using SpeechRecognition and save to a file."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            with open(speech_log_file, "a") as speech_file:
                speech_file.write(f"{time.time()}: {text}\n")
            print(f"Recognized speech: {text}")
        except Exception as e:
            print(f"Speech recognition failed: {e}")

def process_video(video_path, audio_path):
    """Process the uploaded video: extract audio, classify actions, and save logs."""
    cap = cv2.VideoCapture(video_path)
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            detect_actions(results, start_time)

            # Draw landmarks (optional for debugging)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    finally:
        cap.release()

    save_logs()

def extract_audio(video_path, output_audio_path):
    """Extract audio from a video."""
    video = mp_editor.VideoFileClip(video_path)
    video.audio.write_audiofile(output_audio_path, codec="pcm_s16le")

    # Convert speech to text
    convert_speech_to_text(output_audio_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = file.filename
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(video_path)

            # Process video
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "extracted_audio.wav")
            extract_audio(video_path, audio_path)
            process_video(video_path, audio_path)

            return redirect(url_for("results", filename=filename))

    return render_template("index.html")

@app.route("/results/<filename>")
def results(filename):
    video_path = url_for("static", filename=f"uploads/{filename}")
    audio_path = url_for("static", filename="uploads/extracted_audio.wav")
    action_log_path = os.path.join(app.config["UPLOAD_FOLDER"], "action_log.txt")
    speech_log_path = os.path.join(app.config["UPLOAD_FOLDER"], "speech_log.txt")

    # Read logs
    action_log = open(action_log_path).read() if os.path.exists(action_log_path) else ""
    speech_log = open(speech_log_path).read() if os.path.exists(speech_log_path) else ""

    return render_template("results.html", video_path=video_path, audio_path=audio_path,
                           action_log=action_log, speech_log=speech_log)

# if __name__ == "__main__":
#     app.run(debug=True)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
