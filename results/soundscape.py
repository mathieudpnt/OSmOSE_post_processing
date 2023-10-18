# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:55:27 2023

@author: torterma
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_PSD(filename : str, c : str):
    
    npz_mat= np.load(filename, allow_pickle = True)
    f=npz_mat['Freq']
    Sxx=npz_mat['LTAS']

    RMSlevel = 10 * np.log10(np.nanmean(10 ** (Sxx.transpose() / 10), axis=0))

    ax.plot(f, RMSlevel, label='RMS level', color = c)











# Name of pkl file containing welch values
filename = 'C:/Users/torterma/Documents/Projets_Osmose/Sciences/2_FirstSoundscapeAnalysis/D1_LTAS_all.npz'

npz_mat= np.load(filename, allow_pickle = True)
f=npz_mat['Freq']
Sxx=npz_mat['LTAS']
t=npz_mat['time']


# list1, list2 = zip(*sorted(zip(t, Sxx)))
# conv_tup = np.array(list2)
# Sxx=20*np.log10(np.transpose(conv_tup)/10e-12)
# t=list1


fig, ax = plt.subplots(figsize=(40,15))
#im=ax.imshow(Sxx[1:-1], aspect='auto',origin = 'lower')
im=ax.imshow(Sxx[1:-1], aspect='auto', extent=[t[0], t[-1], f[1], f[-1]],origin = 'lower', vmin = 40, vmax = 80)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.tick_params(axis='y', rotation=0, labelsize=20)
ax.tick_params(axis='x', rotation=60, labelsize=15)
ax.set_ylabel('Frequency (Hz)', fontsize=30)
ax.set_xlabel('Date', fontsize=30)

cbar = fig.colorbar(im)
cbar.ax.tick_params(labelsize=30)
cbar.ax.set_ylabel('dB ref 1µPa/Hz', rotation=270, fontsize = 30, labelpad = 40)



# Compute noise in frequency band

fmin = 62.5
fmax = 625
Sxx_fin = Sxx[160,:]
Sxx_fmax = Sxx[500,:]

fig, ax = plt.subplots(figsize=(30,15))
plt.plot(t, Sxx_fin)
plt.plot(t, Sxx_fmax)

X = np.fft.fft(Sxx_fmax)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, 20*np.log10(np.abs(X)), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
#plt.xlim(0, 10)

plt.subplot(122)
plt.plot(t, np.fft.ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()





# Compute PSD RMS


fig, ax = plt.subplots(figsize=(30,15))
ax.grid(True)
ax.set_xlabel('Frequence (Hz)', fontsize=15)
ax.set_ylabel('Densité Spectrale de Puissance (µPa²/Hz)', fontsize=15)
ax.set_xscale('log')

plot_PSD(filename, c='orange')
plot_PSD('C:/Users/torterma/Documents/Projets_Osmose/Sciences/2_FirstSoundscapeAnalysis/E1_LTAS_all.npz', c = 'blue')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['site D phase 1', 'site E phase 1'])
ax.set_xlim(0,60000)





# # Compute PSD percentiles

percen = [1, 5, 50, 95, 99]
p = np.nanpercentile(Sxx.transpose(), percen, 0, interpolation='linear')





for i in range(len(p)):
    plt.plot(f, p[i, :], linewidth=2, label='%s %% percentil' % percen[i])
            










