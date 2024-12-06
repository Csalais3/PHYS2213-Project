import numpy as np
import librosa
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Load the audio file
audio_file = 'recordedFile.wav'  # Replace with desired file path
audio_data, sample_rate = librosa.load(audio_file, sr=None)

# Performs FFT on the audio data
fft_data = np.fft.fft(audio_data)

# Gets the frequency axis
freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)

# Performs the Inverse FFT to recover the time-domain signal
reconstructed_signal = np.fft.ifft(fft_data)

# Converts to real values (discarding imaginary component)
reconstructed_signal = np.real(reconstructed_signal)

# Normalizes the signal (prevents clipping when saving as .wav)
reconstructed_signal = np.int16(reconstructed_signal / np.max(np.abs(reconstructed_signal)) * 32767)

# Saves the reconstructed audio as a .wav file
output_file = 'reconstructed_audio.wav'  # Output file name
wav.write(output_file, sample_rate, reconstructed_signal)

print(f"Reconstructed audio saved as {output_file}")

# Plots the time-domain waveform of the original and reconstructed signals
# Time axis for plotting
time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))

# Plots the original signal's waveform
plt.figure(figsize=(12, 6))

# Plots original audio signal
plt.subplot(2, 1, 1)
plt.plot(time, audio_data)
plt.title("Original Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Plots the reconstructed audio signal's waveform
plt.subplot(2, 1, 2)
plt.plot(time, reconstructed_signal)
plt.title("Reconstructed Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

