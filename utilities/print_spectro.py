import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import os

# Load the WAV file
# path = 'Y:/Bioacoustique/APOCADO2/Campagne 6/PASSE PARTOUT/bouts rouges/7178/analysis/C6D3/results/3_96000 wav'
path = 'C:/Users/dupontma2/Desktop'
sample_rate, audio_data = wavfile.read(os.path.join(path, '2023_06_05_14_10_00.wav'))



# Parameters
nfft = 1024
window_size = 1024
overlap = 20

# Calculate the overlap in samples
overlap_samples = int(overlap / 100 * window_size)

# Generate the spectrogram
frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate, nperseg=window_size, noverlap=overlap_samples, nfft=nfft)

# Plot the spectrogram
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx))  # Convert to dB scale for visualization
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.colorbar(label='dB')
plt.show()

Nbech = len(audio_data)
size_x = (Nbech - window_size) / overlap_samples
size_y = nfft / 2

print(f'\nX: {size_x}\nY: {size_y}')

