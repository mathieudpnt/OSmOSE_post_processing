import glob
import numpy as np
import os
from tqdm import tqdm
import soundfile as sf


# %% Files list

lst_fichiers = sorted(glob.glob('C:/Users/dupontma2/Desktop/extraits sonores/*.wav'))


# %%
for file in tqdm(lst_fichiers):

    data, fs = sf.read(file)

    normalized = np.where(data > 0, data / data.max(), np.where(data < 0, -data / data.min(), data))
    data_norm = np.transpose(np.array([(data / np.max(np.abs(data)))]))[:, 0]
    new_fn = os.path.basename(file).split('.wav')[0] + '_boosted.wav'
    new_path = os.path.join(os.path.dirname(file), new_fn)
    sf.write(file=new_path, data=data_norm, samplerate=fs)
