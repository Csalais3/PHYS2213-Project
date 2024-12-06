import numpy as np
import librosa
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Load the audio file
audio_file = 'recordedFile.wav'  # Replace with your file path
audio_data, sample_rate = librosa.load(audio_file, sr=None)

# Perform FFT on the audio data
fft_data = np.fft.fft(audio_data)

# Get the frequency axis
freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)

# Modify the frequency components if necessary (e.g., filtering, amplification)
# In this case, let's just keep the frequencies as is

# Perform the Inverse FFT to recover the time-domain signal
reconstructed_signal = np.fft.ifft(fft_data)

# Convert to real values (discard imaginary part)
reconstructed_signal = np.real(reconstructed_signal)

# Normalize the signal (to prevent clipping when saving as .wav)
reconstructed_signal = np.int16(reconstructed_signal / np.max(np.abs(reconstructed_signal)) * 32767)

# Save the reconstructed audio as a .wav file
output_file = 'reconstructed_audio.wav'  # Output file name
wav.write(output_file, sample_rate, reconstructed_signal)

print(f"Reconstructed audio saved as {output_file}")

# Plot the time-domain waveform of the original and reconstructed signals
# Time axis for plotting
time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))

# Plot the original signal's waveform
plt.figure(figsize=(12, 6))

# Plot original audio signal
plt.subplot(2, 1, 1)
plt.plot(time, audio_data)
plt.title("Original Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Plot the reconstructed audio signal's waveform
plt.subplot(2, 1, 2)
plt.plot(time, reconstructed_signal)
plt.title("Reconstructed Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

