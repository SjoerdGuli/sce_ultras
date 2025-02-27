import os
import openai
import sounddevice as sd
import numpy as np
import wave
import time
import threading
from dotenv import load_dotenv
from gtts import gTTS
from playsound import playsound

import tkinter as tk
from PIL import Image, ImageTk

# Setup: load API key and client
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is missing in the .env file.")
    exit()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Audio settings and directories
SAMPLE_RATE = 16000  # Whisper requires 16kHz audio
CHANNELS = 1
FILE_NAME = "temp_input.wav"
AUDIO_CACHE_DIR = os.path.join(os.getcwd(), "audio_cache")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Global chat history string
conversation_history = ""

def record_audio():
    audio_data = []
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16)
        stream.start()
    except Exception as e:
        print("Audio stream error:", e)
        return False

    duration = 3  # seconds
    end_time = time.time() + duration
    while time.time() < end_time:
        try:
            audio_chunk, _ = stream.read(1024)
            audio_data.append(audio_chunk)
        except Exception as e:
            print("Error reading stream:", e)
            break
        time.sleep(0.01)

    stream.stop()
    stream.close()

    if not audio_data:
        return False

    try:
        audio_array = np.concatenate(audio_data, axis=0)
    except Exception:
        return False

    try:
        with wave.open(FILE_NAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 2 bytes per sample for int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_array.tobytes())
    except Exception as e:
        print("Error writing WAV:", e)
        return False

    return True

def transcribe_audio():
    try:
        with open(FILE_NAME, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        transcript = response.text.strip()
        return transcript
    except Exception as e:
        print("Transcription error:", e)
        return None

def generate_text_response(prompt):
    try:
        messages = [
            {"role": "system", "content": (
                "You are a robot in an elderly home with people with dementia. "
                "The elderly person thinks you are a relative, so talk like you are a relative. "
                "Try to boost the mood of the elderly person. "
                "Keep the response short so the elderly person can talk more. "
                "Do not hallucinate or say things you don't know."
            )},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        text_response = response.choices[0].message.content.strip()
        return text_response
    except Exception as e:
        print("Chat generation error:", e)
        return None

def generate_and_play_speech(text):
    try:
        tts = gTTS(text=text, lang="en")
        speech_file = os.path.join(AUDIO_CACHE_DIR, "ai_speech.mp3")
        tts.save(speech_file)
        playsound(speech_file)
        os.remove(speech_file)
    except Exception as e:
        print("TTS error:", e)

def record_and_process(ui_callback):
    global conversation_history
    if not record_audio():
        ui_callback("Error recording audio.\n", sender="system")
        return
    transcript = transcribe_audio()
    if transcript:
        conversation_history += f"User: {transcript}\n"
        ui_callback(transcript, sender="user")
        response_text = generate_text_response(conversation_history)
        if response_text:
            conversation_history += f"{response_text}\n"
            ui_callback(response_text, sender="ai")
            generate_and_play_speech(response_text)
    else:
        ui_callback("Could not transcribe audio.\n", sender="system")

class VoiceChatUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fullscreen Chat (30% right)")

        # Make window fullscreen
        self.attributes("-fullscreen", True)

        # Screen dimensions
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        self.recording_in_progress = False

        # Left Canvas (70% width, blank)
        self.left_canvas_width = int(self.screen_width * 0.7)
        self.left_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.left_canvas.place(x=0, y=0,
                               width=self.left_canvas_width,
                               height=self.screen_height)

        # Right Canvas (30% width) for the chat
        self.right_canvas_width = int(self.screen_width * 0.3)
        self.right_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.right_canvas.place(x=self.left_canvas_width, y=0,
                                width=self.right_canvas_width,
                                height=self.screen_height)

        # We'll manually place the header at the top of the right canvas
        self.header_y = 10
        self.draw_header()

        # We'll place the next chat message starting below the header
        self.message_y = 100

        # Create a "Record" button near the bottom center of the right canvas
        self.record_btn = tk.Button(self, text="Record (or press SPACE)",
                                    command=self.start_recording)
        # We'll place it once the window is loaded
        self.after(100, self.place_record_button)

        # Bind spacebar to trigger recording
        self.bind("<space>", lambda event: self.start_recording())
        # Allow ESC to exit fullscreen
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

    def draw_header(self):
        # Profile image on the right canvas
        self.profile_img_path = "./static/sjoerd.jpg"
        if os.path.exists(self.profile_img_path):
            profile_image = Image.open(self.profile_img_path)
            profile_image = profile_image.resize((60, 60), Image.LANCZOS)
            self.profile_photo = ImageTk.PhotoImage(profile_image)
            self.right_canvas.create_image(10, self.header_y,
                                           anchor="nw",
                                           image=self.profile_photo)
        else:
            # If no image, just draw text "No image"
            self.right_canvas.create_text(10, self.header_y,
                                          anchor="nw",
                                          text="No image",
                                          font=("Helvetica", 12, "bold"))

        # Name label "Sjoerd" next to the picture
        self.right_canvas.create_text(80, self.header_y + 20,
                                      anchor="nw",
                                      text="Sjoerd",
                                      font=("Helvetica", 16, "bold"))

    def place_record_button(self):
        # Place the record button near the bottom center of the RIGHT canvas
        x_center = self.left_canvas_width + (self.right_canvas_width // 2)
        y_bottom = self.screen_height - 50
        self.record_btn.place(x=x_center, y=y_bottom, anchor="center")

    def add_message(self, message, sender):
        """
        Draw messages in the right canvas:
          - AI: aligned left, near x=10
          - User: aligned right, near x=(right_canvas_width - 20) for extra margin
        We set 'width' in create_text to automatically wrap lines.
        """

        # Decide alignment and color
        if sender == "ai":
            text_str = f"{message}"
            x_pos = 10
            anchor_style = "nw"
            bubble_bg = "#e1ffc7"
        elif sender == "user":
            text_str = f"You: {message}"
            # Move 20 px from the right edge for margin
            x_pos = self.right_canvas_width - 20
            anchor_style = "ne"
            bubble_bg = "#c7d8ff"
        else:
            text_str = message
            x_pos = self.right_canvas_width // 2
            anchor_style = "n"
            bubble_bg = "#ffd7d7"

        # We will set a max bubble width so text wraps automatically
        max_bubble_width = int(self.right_canvas_width * 0.8)

        # 1) Create text with final arguments (including 'width' for wrapping).
        temp_id = self.right_canvas.create_text(
            x_pos,
            self.message_y,
            text=text_str,
            font=("Helvetica", 12),
            anchor=anchor_style,
            width=max_bubble_width
        )
        bbox = self.right_canvas.bbox(temp_id)  # (x1, y1, x2, y2)
        if not bbox:
            return  # Safety check, no bounding box found
        x1, y1, x2, y2 = bbox

        # 2) Remove that text so we can draw a bubble behind it
        self.right_canvas.delete(temp_id)

        # Expand bounding box for bubble padding
        padding = 10
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

        # 3) Draw the bubble rectangle
        self.right_canvas.create_rectangle(
            x1, y1, x2, y2,
            fill=bubble_bg,
            outline=bubble_bg,
            width=2
        )

        # 4) Redraw the text on top, same arguments
        self.right_canvas.create_text(
            x_pos,
            self.message_y,
            text=text_str,
            font=("Helvetica", 12),
            anchor=anchor_style,
            width=max_bubble_width
        )

        # 5) Move down for next message
        message_height = (y2 - y1)
        self.message_y += message_height + 10

    def update_chat_history(self, message, sender="system"):
        self.add_message(message, sender)

    def start_recording(self):
        if self.recording_in_progress:
            return
        self.recording_in_progress = True
        threading.Thread(target=self.process_voice).start()

    def process_voice(self):
        record_and_process(self.update_chat_history)
        self.recording_in_progress = False

if __name__ == "__main__":
    app = VoiceChatUI()
    app.mainloop()
