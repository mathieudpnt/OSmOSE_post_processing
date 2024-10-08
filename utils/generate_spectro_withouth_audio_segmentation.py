import librosa
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime as dt
import sox
import os

start = dt.now()

file = r"Y:\Bioacoustique\APOCADO2\Campagne 7\7190\flac\7190.230412134430.flac"
out_folder = r"Y:\Bioacoustique\APOCADO2\Campagne 7\7190\flac"

sr = 128000


audio_data = librosa.load(path=file, sr=sr, mono=True, offset=2600, duration=10)

# Parameters
nfft = 1024
window_size = 1024
overlap = 20

# Calculate the overlap in samples
overlap_samples = int(overlap / 100 * window_size)

# Generate the spectrogram
frequencies, times, Sxx = spectrogram(
    audio_data[0], fs=sr, nperseg=window_size, noverlap=overlap_samples, nfft=nfft
)

# Plot the spectrogram
plt.pcolormesh(
    times, frequencies, 10 * np.log10(Sxx)
)  # Convert to dB scale for visualization
plt.xticks([], [])
plt.yticks([], [])
plt.axis("off")
plt.subplots_adjust(
    top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
)  # delete white borders

# plt.show()
plt.savefig(os.path.join(out_folder, "test.png"), bbox_inches="tight", pad_inches=0)
stop = dt.now()
elapsed = (stop - start).total_seconds()


print(f"Elapsed time: {elapsed:.2f}s")
