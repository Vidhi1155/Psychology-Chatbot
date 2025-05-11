from flask import Flask, render_template, request, jsonify
import requests
import speech_recognition as sr
import pyttsx3
import threading

flask_app = Flask(__name__)
FASTAPI_URL = "http://localhost:8001"

engine = pyttsx3.init()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/send", methods=["POST"])
def send():
    try:
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        response = requests.post(f"{FASTAPI_URL}/chat", json={"query": user_input})
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route("/speech_to_text", methods=["GET"])
def speech_to_text():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
        text = recognizer.recognize_google(audio)
        return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"text": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"text": f"Speech recognition error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"text": f"Error: {str(e)}"}), 500

@flask_app.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    try:
        text = request.json.get("text")
        if not text:
            return jsonify({"error": "Text is required"}), 400

        threading.Thread(target=speak_text, args=(text,)).start()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route("/get_memory", methods=["GET"])
def get_memory():
    try:
        response = requests.get(f"{FASTAPI_URL}/memory")
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    flask_app.run(debug=True, port=5000)