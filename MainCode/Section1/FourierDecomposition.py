import numpy as np
import librosa
import matplotlib.pyplot as plt

def execute():
    # Loads the audio file
    audio_file = 'recordedFile.wav'
    audio_data, sample_rate = librosa.load(audio_file, sr=None)

    # Performs FFT on the audio data
    fft_data = np.fft.fft(audio_data)

    # Gets the frequency axis
    freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)

    # Calculates the magnitude
    magnitude = np.abs(fft_data)

    # Keeps positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]

    # Plots the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, positive_magnitude)
    plt.title('Frequency Spectrum of Audio')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    
    return

if __name__ == "__main__":
    execute()