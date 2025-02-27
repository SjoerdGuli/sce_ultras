import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

# ----- Load Audio -----
audio_file = "ai_speech.mp3"  # Replace with your actual file path
# Load audio as mono at 44100 Hz
y, sr = librosa.load(audio_file, sr=44100, mono=True)
audio_index = 0  # Global index for audio playback

# Global modulation variable (updated in the audio callback)
current_modulation = None

# ----- Parameters -----
num_points = 200         # Total number of points around the circle
buffer_size = 2048       # Block size for audio processing
frame_rate = 120          # Frames per second for the animation
amplitude_scale = 0.2    # Scaling factor for the modulation amplitude

# Generate theta values for the full circle (0 to 2π)
theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

# Base circle: radius 1 for each angle
def get_circle(mod):
    # mod is an array of length num_points
    radius = 1 + mod
    x = np.cos(theta) * radius
    y_vals = np.sin(theta) * radius
    return np.column_stack((x, y_vals))

# ----- Matplotlib Setup -----
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axis("off")
# Create an initial filled polygon (circle with no modulation)
init_xy = get_circle(np.zeros(num_points))
polygon = Polygon(init_xy, closed=True, facecolor="dodgerblue", edgecolor="none", alpha=0.8)
ax.add_patch(polygon)

def smooth_signal(signal, window_len=20):
    """Apply a simple moving average filter with a longer window to smooth the signal."""
    return np.convolve(signal, np.ones(window_len)/window_len, mode='same')

# ----- Audio Callback -----
def audio_callback(outdata, frames, time, status):
    global audio_index, current_modulation, y
    if status:
        print(status)
    end_index = audio_index + frames
    if end_index > len(y):
        outdata.fill(0)
        return

    # Play the next chunk of audio
    chunk = y[audio_index:end_index]
    outdata[:, 0] = chunk
    audio_index += frames

    # --- Process the audio chunk for modulation ---
    # Resample the chunk to have num_points values
    x_old = np.linspace(0, 1, len(chunk))
    x_new = np.linspace(0, 1, num_points)
    resampled = np.interp(x_new, x_old, chunk)
    
    # Smooth the resampled signal with a longer window to reduce spikes
    smoothed = smooth_signal(resampled, window_len=20)
    
    # Remove the DC offset (center the modulation around zero)
    mod = smoothed - np.mean(smoothed)
    
    # Normalize the modulation to [-1, 1]
    if np.max(np.abs(mod)) > 0:
        mod = mod / np.max(np.abs(mod))
    else:
        mod = np.zeros_like(mod)
    
    # Scale the modulation effect
    current_modulation = mod * amplitude_scale

# ----- Animation Update Function -----
def update(frame):
    global current_modulation
    # Use the current modulation; if not available, default to zeros.
    mod = current_modulation if current_modulation is not None else np.zeros(num_points)
    # Update the polygon's vertices for the full circle with modulation
    new_xy = get_circle(mod)
    polygon.set_xy(new_xy)
    return polygon,

# ----- Start Audio Playback and Animation -----
stream = sd.OutputStream(callback=audio_callback, samplerate=sr, channels=1, blocksize=buffer_size)
stream.start()

ani = animation.FuncAnimation(fig, update, interval=1000/frame_rate, blit=True)
plt.show()

stream.stop()
