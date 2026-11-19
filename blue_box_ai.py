import time
import wave
import numpy as np
import requests
import sounddevice as sd
import keyboard
import pyttsx3
from pywhispercpp.model import Model

print("🔵 Booting Blue Box Offline AI (Whisper Engine)...")

# Pehli baar mein 142MB 'base.en' model auto-download hoga, uske baad 100% offline
whisper_model = Model('small', print_realtime=False, print_progress=False)

SAMPLE_RATE = 16000
WAV_FILE = "input.wav"

SYSTEM_PROMPT = (
    "You are Blue Box, a friendly, screen-free pocket study tutor for Indian students. "
    "Understand English, Hindi, and Hinglish questions. Reply in the same language style the student uses (simple English or easy Hinglish using Roman script so text-to-speech reads it clearly). "
    "Keep answers accurate, factual, and under 3 to 4 short sentences. "
    "If a question is unclear or noise, say: Sorry, I did not catch that clearly. Please repeat your question."
)

def speak(text):
    clean_text = text.replace("**", "").replace("*", "").replace("#", "").strip()
    print(f"\n🔊 Blue Box: {clean_text}\n")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    chosen = None
    for key in ["heera", "ravi", "india", "hazel", "zira"]:
        for v in voices:
            if key in v.name.lower():
                chosen = v.id
                break
        if chosen:
            break
    if chosen:
        engine.setProperty('voice', chosen)
    engine.setProperty('rate', 148)
    engine.setProperty('volume', 1.0)
    engine.say(clean_text)
    engine.runAndWait()
    engine.stop()

def record_and_transcribe():
    print("🟡 Hold [SPACEBAR] to ask your question... (Press ESC to quit)")
    while True:
        if keyboard.is_pressed('esc'):
            return None
        if keyboard.is_pressed('space'):
            break
        time.sleep(0.05)

    print("🎙️ Listening... (Speak clearly & release SPACEBAR when done)")
    frames = []

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while keyboard.is_pressed('space'):
            data, _ = stream.read(1024)
            frames.append(data.copy())

    if not frames:
        return ""

    audio_data = np.concatenate(frames, axis=0).astype(np.float32)
    peak = np.max(np.abs(audio_data))
    audio_data = (audio_data / peak * 28000).astype(np.int16) if peak > 0 else audio_data.astype(np.int16)

    with wave.open(WAV_FILE, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    prompt_hint = 'Indian student asking in English or Hinglish: Read Only Memory kya hota hai, ROM, RAM, CPU, Operating System, Photosynthesis, NCERT Science.'
    segments = whisper_model.transcribe(WAV_FILE, initial_prompt=prompt_hint)
    text = " ".join([seg.text for seg in segments]).strip()
    if any(tag in text.lower() for tag in ["[blank_audio]", "(silence)", "[music]", "[applause]", "[noise]"]) or len(text) < 3:
        return ""
    return text

def ask_offline_brain(question):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:1.5b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 150}
        }
    )
    return response.json()["message"]["content"].strip()

speak("Blue Box is ready with Whisper voice engine. Hold spacebar and ask your question.")

while True:
    user_text = record_and_transcribe()
    if user_text is None:
        speak("Shutting down Blue Box. Goodbye!")
        break
    if not user_text:
        speak("I didn't hear anything. Please hold spacebar and try again.")
        continue

    print(f"🗣️ You asked: {user_text}")
    print("🧠 Thinking offline...")
    try:
        answer = ask_offline_brain(user_text)
        speak(answer)
    except Exception as e:
        print("Error:", e)
        speak("Please make sure Ollama is running in the background.")