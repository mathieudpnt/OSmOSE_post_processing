from pydub import AudioSegment
import os
from datetime import datetime as dt

input_directory = r"Y:\Bioacoustique\APOCADO2\Campagne 7\7190\wav"
output_directory = r"Y:\Bioacoustique\APOCADO2\Campagne 7\7190\flac"

os.makedirs(output_directory, exist_ok=True)

for filename in os.listdir(input_directory)[168]:
    if filename.endswith(".wav"):
        start = dt.now()
        wav_path = os.path.join(input_directory, filename)
        flac_path = os.path.join(
            output_directory, os.path.splitext(filename)[0] + ".flac"
        )

        audio = AudioSegment.from_wav(wav_path)
        audio.export(flac_path, format="flac")
        stop = dt.now()
        elapsed = (stop - start).total_seconds()
        print(f"Converted {wav_path} to {flac_path}, elapsed time: {elapsed:.0f} s")
