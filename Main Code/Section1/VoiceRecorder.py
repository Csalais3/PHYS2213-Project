import pyaudio
import wave
import time
import threading
import numpy as np
import matplotlib.pyplot as plt


# Constants for audio processing
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 2048
OUTPUT_FILENAME = 'recordedFile.wav'

# Initialing PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format = FORMAT, channels= CHANNELS, rate= RATE, input= True, frames_per_buffer= CHUNK)
frames = []

print("Press Enter to start recording...")
input()
time.sleep(0.2)

stop_event = threading.Event()

def stop_on_input():
    input("Press Enter to stop recording...")
    stop_event.set()

# Start a separate thread for stopping the recording
thread = threading.Thread(target=stop_on_input)
thread.start()

print("Recording... Press Enter to stop.")
while not stop_event.is_set():
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

# Closes the audio stream
stream.stop_stream()
stream.close()
audio.terminate()

# Saves the reconstructed audio as a .wav file
waveFile = wave.open(OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))

print("Recording saved to", OUTPUT_FILENAME)

audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

# Plot the audio waveform
plt.figure(figsize=(10, 4))
plt.plot(audio_data)
plt.title("Audio Waveform")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.show()
