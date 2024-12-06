import pyaudio
import time
import matplotlib.pyplot as plt
import numpy as np
import threading
import struct

# Constants for audio processing
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 2048
OUTPUT_FILENAME = 'recordedFile.wav'

# Initialing PyAudio
audio = pyaudio.PyAudio()
info = audio.get_default_input_device_info()
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

# Initialize the figure for plotting
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Bar plot (first plot)
bars = ax1.bar([str(i) for i in range(100)], height=[0] * 100)
ax1.set_title('Audio Amplitude (Bars)')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Amplitude')

# Frequency spectrum plot (second plot)
line, = ax2.plot([], [], lw=2)  # Line object for the frequency spectrum
ax2.set_title('Frequency Spectrum of Audio')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')
ax2.set_xlim(0, RATE // 2)  # Frequency axis from 0 to Nyquist frequency
ax2.set_ylim(0, 1000)  # Adjust the y-axis range for better visualization

print("Recording... Press Enter to stop.")
while not stop_event.is_set():
    data = stream.read(CHUNK, exception_on_overflow=False)
    read_data = np.array(struct.unpack(f'{CHUNK}h', data))[0:CHUNK]
    x_values = range(len(read_data))  # X-axis positions for the bars
    for i, bar in enumerate(bars):
        bar.set_height(read_data[i])  # Update the bar heights

    # Dynamically adjust the y-axis limits to fit the amplitude range
    min_value = np.min(read_data)
    max_value = np.max(read_data)
    if max_value > min_value:
        ax1.set_ylim(min_value * 1.1, max_value * 1.1)
    else:
        ax1.set_ylim(-1, 1)  # Default y-axis range

    # Reduce the bar width for smoother appearance
    for bar in bars:
        bar.set_width(0.5)  # Adjust bar width as needed (smaller value = narrower bars)
    
    # Perform FFT on the audio data
    fft_data = np.fft.fft(read_data)  

    # Get the frequency axis
    freqs = np.fft.fftfreq(len(fft_data), 1 / RATE)

    # Calculate the magnitude (absolute value)
    magnitude = np.abs(fft_data)

    # Only keep positive frequencies
    positive_freqs = freqs[:len(freqs) // 2]
    positive_magnitude = magnitude[:len(magnitude) // 2]

    # Normalize the magnitude for better visualization
    positive_magnitude = np.log(positive_magnitude + 1e-10)

    # Update the frequency spectrum plot with new data
    line.set_data(positive_freqs, positive_magnitude)
    
    # Adjust the y-axis limit dynamically based on the magnitude
    ax2.set_ylim(0, np.max(positive_magnitude) + 1)

    # Redraw the plots
    fig.canvas.draw()
    fig.canvas.flush_events()

# Closes the audio stream
stream.stop_stream()
stream.close()
audio.terminate()