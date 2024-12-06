import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct

def execute():
    # Constants for audio processing
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    CHUNK = 2048

    # Initialing PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Initializing the plot
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))  # Create 4 subplots (3 individual + 1 combined)
    time = np.linspace(0, CHUNK / RATE, CHUNK)  # Time axis for one CHUNK
    lines = [ax.plot([], [])[0] for ax in axs]  # Create line objects for each subplot

    # Sets up axes for the subplots
    for i, ax in enumerate(axs):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0, CHUNK / RATE)
        ax.set_ylim(-1, 1)  # Default y-axis range, adjusted dynamically later
        ax.grid(True)

    axs[0].set_title("Reconstructed Sinusoid 1 (Dominant Frequency)")
    axs[1].set_title("Reconstructed Sinusoid 2 (Second Dominant Frequency)")
    axs[2].set_title("Reconstructed Sinusoid 3 (Third Dominant Frequency)")
    axs[3].set_title("Combined Wave (All Sinusoids)")

    # Function to classify the wave type
    def classify_wave(phase):
        if np.isclose(phase, 0, atol=1e-2):
            return "Cosine"
        elif np.isclose(phase, np.pi / 2, atol=1e-2) or np.isclose(phase, -np.pi / 2, atol=1e-2):
            return "Sine"
        else:
            return "Shifted Sine"

    print("Press Ctrl+C to stop recording...")
    try:
        while True:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.array(struct.unpack(f'{CHUNK}h', data))  # Convert to numpy array

            # Perform FFT
            fft_data = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft_data), 1 / RATE)

            # Get the positive frequencies
            positive_freqs = freqs[:len(freqs) // 2]
            positive_fft = fft_data[:len(fft_data) // 2]

            # Identify dominant frequencies (top 3 by magnitude)
            magnitudes = np.abs(positive_fft)
            top_indices = np.argsort(magnitudes)[-3:]  # Indices of top 3 frequencies
            dominant_frequencies = positive_freqs[top_indices]
            dominant_magnitudes = magnitudes[top_indices]
            dominant_phases = np.angle(positive_fft[top_indices])

            # Clear and update each subplot with a sinusoid
            combined_wave = np.zeros(CHUNK)

            for i, (f, A, phi) in enumerate(zip(dominant_frequencies, dominant_magnitudes, dominant_phases)):
                # Reconstruct the sinusoid
                sinusoid = A * np.sin(2 * np.pi * f * time + phi)

                # Classify wave based on phase angle
                wave_type = classify_wave(phi)

                # Plot individual sinusoids
                axs[i].clear()
                axs[i].plot(time, sinusoid, label=f"{wave_type}: f = {f:.1f} Hz")
                axs[i].set_title(f"Reconstructed Sinusoid {i + 1}")
                axs[i].set_xlabel("Time (s)")
                axs[i].set_ylabel("Amplitude")
                axs[i].set_xlim(0, CHUNK / RATE)
                axs[i].set_ylim(-A * 1.1, A * 1.1)  # Dynamically adjust y-axis
                axs[i].legend()

                # Add sinusoid to combined wave
                combined_wave += sinusoid

            # Plot the combined wave
            axs[3].clear()
            axs[3].plot(time, combined_wave, label="Combined Wave (All Components)")
            axs[3].set_title("Combined Wave (All Sinusoids)")
            axs[3].set_xlabel("Time (s)")
            axs[3].set_ylabel("Amplitude")
            axs[3].set_xlim(0, CHUNK / RATE)
            axs[3].set_ylim(-np.max(np.abs(combined_wave)) * 1.1, np.max(np.abs(combined_wave)) * 1.1)
            axs[3].legend()

            # Update plots
            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("Stopped.")

    # Closes the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == "__main__":
    execute()