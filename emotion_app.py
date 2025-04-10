import os
# Suppress TensorFlow/Keras logging (this removes the 1/1 ━━━ logging)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from keras.models import load_model
import openai
import sounddevice as sd
import wave
import time
import threading
from dotenv import load_dotenv
from gtts import gTTS
import tkinter as tk
from PIL import Image, ImageTk
import librosa  # for loading audio

# ----------------------------
# Helper function for emotion mapping
# ----------------------------
def map_emotion(label):
    mapping = {
        "Angry": "not happy",
        "Disgust": "not happy",
        "Fear": "not happy",
        "Sad": "not happy",
        "Happy": "happy",
        "Surprise": "happy",
        "Neutral": None
    }
    return mapping.get(label, None)

# ----------------------------
# Global Audio Variables for Playback
# ----------------------------
y = None       # Audio data array
sr = None      # Sample rate
audio_index = 0  # Playback index

# Global variable for tracking dominant emotion
current_dominant_emotion = "Neutral"
# Global bit value for therapy mode: 0 means reminiscence therapy, 1 means small talk.
therapy_mode = 0  
# Global counter for continuous duration (in seconds) when dominant emotion is 'not happy'
not_happy_duration = 0

def audio_callback(outdata, frames, time_info, status):
    global audio_index, y
    if status:
        print(status)
    end_index = audio_index + frames
    if y is None or end_index > len(y):
        outdata.fill(0)
        return
    chunk = y[audio_index:end_index]
    outdata[:, 0] = chunk
    audio_index += frames

def play_audio_file(speech_file):
    try:
        # Load the audio at a fixed sample rate (22050 Hz) for consistency.
        audio_data, sample_rate = librosa.load(speech_file, sr=22050, mono=True)
        print("[DEBUG] Loaded audio with sample rate:", sample_rate, "and length:", len(audio_data))
    except Exception as e:
        print("Error loading audio:", e)
        return

    playback_index = 0  # Local playback index

    def local_callback(outdata, frames, time_info, status):
        nonlocal playback_index
        if status:
            print(status)
        end_index = playback_index + frames
        if end_index > len(audio_data):
            outdata[:len(audio_data) - playback_index, 0] = audio_data[playback_index:]
            outdata[len(audio_data) - playback_index:, 0] = 0
            playback_index = len(audio_data)
        else:
            outdata[:, 0] = audio_data[playback_index:end_index]
            playback_index += frames

    with sd.OutputStream(callback=local_callback, samplerate=sample_rate, channels=1):
        while playback_index < len(audio_data):
            time.sleep(0.1)
    print("[DEBUG] Audio playback completed.")

# ----------------------------
# Emotion Detection Component (Video)
# ----------------------------
class EmotionDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[DEBUG] Error: Could not open video stream from camera.")
            exit()
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("[DEBUG] Loading emotion recognition model...")
        self.emotion_classifier = load_model("./emotion_recognition/emotion_model.hdf5", compile=False)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        print("[DEBUG] Emotion recognition model loaded.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        detected_emotions = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            try:
                roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print("[DEBUG] Error resizing face region:", e)
                continue
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            preds = self.emotion_classifier.predict(roi)[0]
            label = self.emotion_labels[preds.argmax()]
            detected_emotions.append(label)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (36, 255, 12), 2)
        return frame, detected_emotions

    def release(self):
        self.cap.release()

# ----------------------------
# Chatbot & Audio Functions
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[DEBUG] Error: OPENAI_API_KEY is missing in the .env file.")
    exit()
client = openai.OpenAI(api_key=OPENAI_API_KEY)

SAMPLE_RATE = 16000  # for recording (Whisper needs 16kHz)
CHANNELS = 1
FILE_NAME = "temp_input.wav"
AUDIO_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

conversation_history = ""

def record_audio():
    print("[DEBUG] Starting audio recording...")
    audio_data = []
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
        stream.start()
    except Exception as e:
        print("[DEBUG] Audio stream error:", e)
        return False
    duration = 3  # seconds (default duration if not using hold mode)
    end_time = time.time() + duration
    while time.time() < end_time:
        try:
            audio_chunk, _ = stream.read(1024)
            audio_data.append(audio_chunk)
        except Exception as e:
            print("[DEBUG] Error reading stream:", e)
            break
        time.sleep(0.01)
    stream.stop()
    stream.close()
    print("[DEBUG] Audio recording finished.")
    if not audio_data:
        print("[DEBUG] No audio data recorded.")
        return False
    try:
        audio_array = np.concatenate(audio_data, axis=0)
    except Exception as e:
        print("[DEBUG] Error concatenating audio data:", e)
        return False
    try:
        with wave.open(FILE_NAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_array.tobytes())
        print(f"[DEBUG] Audio written to {FILE_NAME}.")
    except Exception as e:
        print("[DEBUG] Error writing WAV:", e)
        return False
    return True

def transcribe_audio():
    print("[DEBUG] Starting transcription...")
    try:
        with open(FILE_NAME, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        transcript = response.text.strip()
        print("[DEBUG] Transcription result:", transcript)
        return transcript
    except Exception as e:
        print("[DEBUG] Transcription error:", e)
        return None

def generate_text_response(prompt):
    print("[DEBUG] Generating text response...")
    global current_dominant_emotion, therapy_mode
    try:
        if therapy_mode == 0:
            system_message = (
                "You are a robot in an elderly home with people with dementia. The elderly person thinks you are a relative. "
                "Engage in a reminiscence exercise by asking for past memories and sharing relatable stories. "
                "Keep the response around 60 characters."
            )
        else:
            system_message = (
                "You are a friendly and caring relative speaking with an elderly person. "
                "Engage in small talk or gently conclude the conversation. "
                "Keep the response around 60 characters."
            )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        text_response = response.choices[0].message.content.strip()
        print("[DEBUG] Text response generated:", text_response)
        return text_response
    except Exception as e:
        print("[DEBUG] Chat generation error:", e)
        return None

def generate_and_play_speech(text):
    print("[DEBUG] Generating speech from text...")
    try:
        tts = gTTS(text=text, lang="en")
        speech_file = os.path.join(AUDIO_CACHE_DIR, "ai_speech.mp3")
        tts.save(speech_file)
        print("[DEBUG] Speech file saved:", speech_file)
        file_size = os.path.getsize(speech_file)
        print("[DEBUG] Speech file size:", file_size, "bytes")
        play_audio_file(speech_file)
        if os.path.exists(speech_file):
            os.remove(speech_file)
            print("[DEBUG] Speech file deleted.")
    except Exception as e:
        print("[DEBUG] TTS error:", e)

def record_and_process(ui_callback):
    global conversation_history
    if not record_audio():
        ui_callback("Error recording audio.\n", sender="system")
        return
    transcript = transcribe_audio()
    if transcript:
        conversation_history += f"{transcript}\n"
        ui_callback(transcript, sender="user")
        response_text = generate_text_response(conversation_history)
        if response_text:
            conversation_history += f"{response_text}\n"
            ui_callback(response_text, sender="ai")
            generate_and_play_speech(response_text)
    else:
        ui_callback("Could not transcribe audio.\n", sender="system")

def classify_message_emotion(text):
    print("[DEBUG] Classifying text emotion for message:", text)
    try:
        messages = [
            {"role": "system", "content": "You are a text emotion classifier. Classify the following message into one of these emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise. Return only the emotion word."},
            {"role": "user", "content": text}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        emotion = response.choices[0].message.content.strip()
        allowed = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        if emotion not in allowed:
            emotion = "Neutral"
        mapped = map_emotion(emotion)
        print("[DEBUG] Classified emotion:", mapped)
        return mapped
    except Exception as e:
        print("[DEBUG] Error classifying text emotion:", e)
        return None

# ----------------------------
# Tkinter UI with Emotion Detection & Voice Chat
# ----------------------------
class VoiceChatUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fullscreen Chat with Emotion Detection")
        self.attributes("-fullscreen", True)
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()
        self.recording_flag = False  # for recording while SPACE is held

        # Right Canvas (30% width) for chat interface
        self.right_canvas_width = int(self.screen_width * 0.3)
        self.chat_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.chat_canvas.place(x=self.screen_width - self.right_canvas_width, y=0,
                               width=self.right_canvas_width, height=self.screen_height)

        # Left side (70% width) for video
        self.left_canvas_width = int(self.screen_width * 0.7)
        self.top_left_height = int(self.screen_height * 0.6)
        self.bottom_left_height = self.screen_height - self.top_left_height

        self.video_canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.video_canvas.place(x=0, y=0, width=self.left_canvas_width, height=self.top_left_height)
        self.video_canvas_image = self.video_canvas.create_image(0, 0, anchor="nw", image=None)

        self.stats_canvas = tk.Canvas(self, bg="gray", highlightthickness=0)
        self.stats_canvas.place(x=0, y=self.top_left_height, width=self.left_canvas_width,
                                height=self.bottom_left_height)
        
        self.override_played = False  # New flag: override sound not yet played

        self.header_y = 10
        self.draw_header()
        self.message_y = 100

        self.record_btn = tk.Button(self, text="Record (hold SPACE)", command=self.manual_record)
        self.after(100, self.place_record_button)

        # Bind key events for SPACE press and release
        self.bind("<KeyPress-space>", self.start_recording)
        self.bind("<KeyRelease-space>", self.stop_recording)
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

        self.emotion_detector = EmotionDetector()
        # Initialize counters for emotion detection
        self.emotion_counter = {"happy": 0, "not happy": 0}
        self.total_emotion_counter = {"happy": 0, "not happy": 0}
        self.message_emotion_counter = {"happy": 0, "not happy": 0}
        self.mode_switch_animation_active = False

        self.update_video()
        self.after(1000, self.update_emotion_stats)  # update every second
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Immediately start with a reminiscence therapy message from the AI
        initial_message = ("I remember the good old days when you used to sit by the window and watch the birds. "
                           "What is one of your fondest memories?")
        self.update_chat_history(initial_message, sender="ai")
        global conversation_history
        conversation_history += f"{initial_message}\n"
        # Schedule playing the initial message after 500ms so the window appears first.
        self.after(500, lambda: generate_and_play_speech(initial_message))

    def on_closing(self):
        self.emotion_detector.release()
        self.destroy()

    def update_video(self):
        frame, detected_emotions = self.emotion_detector.get_frame()
        if frame is not None:
            for emotion in detected_emotions:
                mapped = map_emotion(emotion)
                if mapped is not None:
                    self.emotion_counter[mapped] += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = frame.shape[:2]
            target_ratio = self.left_canvas_width / self.top_left_height
            frame_ratio = orig_w / orig_h
            if frame_ratio > target_ratio:
                new_w = int(orig_h * target_ratio)
                x_offset = (orig_w - new_w) // 2
                cropped = frame[:, x_offset:x_offset+new_w]
            elif frame_ratio < target_ratio:
                new_h = int(orig_w / target_ratio)
                y_offset = (orig_h - new_h) // 2
                cropped = frame[y_offset:y_offset+new_h, :]
            else:
                cropped = frame
            img = Image.fromarray(cropped)
            img = img.resize((self.left_canvas_width, self.top_left_height))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_imgtk = imgtk
            self.video_canvas.itemconfig(self.video_canvas_image, image=imgtk)
        self.after(30, self.update_video)

    def update_emotion_stats(self):
        global current_dominant_emotion, therapy_mode, not_happy_duration
        # Accumulate counts from the current interval
        for emotion, count in self.emotion_counter.items():
            self.total_emotion_counter[emotion] += count
        sum_face = sum(self.total_emotion_counter.values())
        sum_message = sum(self.message_emotion_counter.values())
        combined_scores = {}
        for emotion in self.emotion_counter.keys():
            norm_face = self.total_emotion_counter[emotion] / sum_face if sum_face > 0 else 0
            norm_message = self.message_emotion_counter[emotion] / sum_message if sum_message > 0 else 0
            combined_scores[emotion] = 0.2 * norm_message + 0.8 * norm_face
        dominant_emotion = max(combined_scores, key=combined_scores.get)
        current_dominant_emotion = dominant_emotion

        # Update the continuous "not happy" counter.
        if dominant_emotion == "not happy":
            not_happy_duration += 1
        else:
            not_happy_duration = 0
            # If desired, reset override flag when mood changes (optional)
            # self.override_played = False

        # Trigger the override sound and red alarm effect only once.
        if not_happy_duration >= 10 and not self.override_played:
            print("[DEBUG] Dominant emotion 'not happy' for 10 seconds. Triggering override.")
            sd.stop()  # Immediately stop any ongoing audio playback.
            override_mp3_path = "./cut_nummer.mp3"
            if os.path.exists(override_mp3_path):
                # Play the override sound in a new thread so the UI remains responsive.
                threading.Thread(target=play_audio_file, args=(override_mp3_path,), daemon=True).start()
            else:
                print("[DEBUG] Override mp3 file not found:", override_mp3_path)
            # Trigger red alarm animation.
            if not self.mode_switch_animation_active:
                self.mode_switch_animation_active = True
                self.animate_mode_switch(remaining=5000, interval=500)
            self.override_played = True  # Set flag so the override plays only once.

        therapy_text = "Reminiscence Therapy" if therapy_mode == 0 else "Small Talk"
        stats_text = f"Dominant Emotion:\n{dominant_emotion}\n\nTherapy Mode:\n{therapy_text}"
        self.stats_canvas.delete("all")
        self.stats_canvas.create_text(self.left_canvas_width//2, self.bottom_left_height//2,
                                    text=stats_text, font=("Helvetica", 24, "bold"),
                                    fill="white", justify="center")
        # Reset the counter for this interval.
        self.emotion_counter = {"happy": 0, "not happy": 0}
        self.after(1000, self.update_emotion_stats)



    def animate_mode_switch(self, remaining, interval):
        # Animate the stats canvas background color (flashing red/darkred)
        current_bg = self.stats_canvas.cget("bg")
        new_bg = "red" if current_bg != "red" else "darkred"
        self.stats_canvas.config(bg=new_bg)
        if remaining > 0:
            self.after(interval, lambda: self.animate_mode_switch(remaining - interval, interval))
        else:
            self.stats_canvas.config(bg="gray")
            self.mode_switch_animation_active = False

    def draw_header(self):
        self.profile_img_path = "./static/sjoerd.jpg"
        if os.path.exists(self.profile_img_path):
            try:
                profile_image = Image.open(self.profile_img_path)
                profile_image = profile_image.resize((60, 60), Image.LANCZOS)
                self.profile_photo = ImageTk.PhotoImage(profile_image)
                self.chat_canvas.create_image(10, self.header_y, anchor="nw", image=self.profile_photo)
            except Exception as e:
                print("Error loading image:", e)
        else:
            self.chat_canvas.create_text(10, self.header_y, anchor="nw",
                                          text="No image", font=("Helvetica", 12, "bold"))
        self.chat_canvas.create_text(80, self.header_y + 20, anchor="nw",
                                      text="Sjoerd", font=("Helvetica", 16, "bold"))

    def place_record_button(self):
        x_center = self.screen_width - (self.right_canvas_width // 2)
        y_bottom = self.screen_height - 50
        self.record_btn.place(x=x_center, y=y_bottom, anchor="center")

    def add_message(self, message, sender):
        if sender == "ai":
            text_str = f"{message}"
            x_pos = 10; anchor_style = "nw"; bubble_bg = "#e1ffc7"
        elif sender == "user":
            text_str = f"{message}"
            x_pos = self.right_canvas_width - 20; anchor_style = "ne"; bubble_bg = "#c7d8ff"
        else:
            text_str = message; x_pos = self.right_canvas_width // 2; anchor_style = "n"; bubble_bg = "#ffd7d7"
        max_bubble_width = int(self.right_canvas_width * 0.8)
        temp_id = self.chat_canvas.create_text(x_pos, self.message_y, text=text_str,
                                                font=("Helvetica", 12), anchor=anchor_style,
                                                width=max_bubble_width)
        bbox = self.chat_canvas.bbox(temp_id)
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        self.chat_canvas.delete(temp_id)
        padding = 10
        x1 -= padding; y1 -= padding; x2 += padding; y2 += padding
        self.chat_canvas.create_rectangle(x1, y1, x2, y2, fill=bubble_bg, outline=bubble_bg, width=2)
        self.chat_canvas.create_text(x_pos, self.message_y, text=text_str,
                                     font=("Helvetica", 12), anchor=anchor_style,
                                     width=max_bubble_width)
        self.message_y += (y2 - y1) + 10

    def update_chat_history(self, message, sender="system"):
        self.add_message(message, sender)
        if sender == "user":
            emotion = classify_message_emotion(message)
            if emotion is not None:
                self.message_emotion_counter[emotion] += 1
        # Reset emotion detection after every robot message.
        if sender == "ai":
            self.emotion_counter = {"happy": 0, "not happy": 0}
            self.total_emotion_counter = {"happy": 0, "not happy": 0}
            self.message_emotion_counter = {"happy": 0, "not happy": 0}
            global not_happy_duration
            not_happy_duration = 0

    def start_recording(self, event):
        if not self.recording_flag:
            print("[DEBUG] SPACE pressed. Starting recording...")
            self.recording_flag = True
            threading.Thread(target=self.record_audio_loop, daemon=True).start()

    def stop_recording(self, event):
        if self.recording_flag:
            print("[DEBUG] SPACE released. Stopping recording...")
            self.recording_flag = False

    def manual_record(self):
        if not self.recording_flag:
            self.recording_flag = True
            threading.Thread(target=self.record_audio_loop, daemon=True).start()

    def record_audio_loop(self):
        print("[DEBUG] Entering audio recording loop...")
        audio_data = []
        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
            stream.start()
        except Exception as e:
            print("[DEBUG] Audio stream error:", e)
            return
        while self.recording_flag:
            try:
                audio_chunk, _ = stream.read(1024)
                audio_data.append(audio_chunk)
            except Exception as e:
                print("[DEBUG] Error reading stream:", e)
                break
            time.sleep(0.01)
        stream.stop()
        stream.close()
        print("[DEBUG] Audio recording loop finished.")
        if not audio_data:
            print("[DEBUG] No audio data recorded in loop.")
            return
        try:
            audio_array = np.concatenate(audio_data, axis=0)
        except Exception as e:
            print("[DEBUG] Error concatenating audio data:", e)
            return
        try:
            with wave.open(FILE_NAME, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_array.tobytes())
            print(f"[DEBUG] Audio written to {FILE_NAME}.")
        except Exception as e:
            print("[DEBUG] Error writing WAV:", e)
            return
        transcript = transcribe_audio()
        if transcript:
            global conversation_history
            conversation_history += f"{transcript}\n"
            self.update_chat_history(transcript, sender="user")
            response_text = generate_text_response(conversation_history)
            if response_text:
                conversation_history += f"{response_text}\n"
                self.update_chat_history(response_text, sender="ai")
                generate_and_play_speech(response_text)
        else:
            self.update_chat_history("Could not transcribe audio.\n", sender="system")
        print("[DEBUG] Recording process completed.")

if __name__ == "__main__":
    app = VoiceChatUI()
    app.mainloop()
