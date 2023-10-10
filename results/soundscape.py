# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:55:27 2023

@author: torterma
"""

import numpy as np
import matplotlib.pyplot as plt

# Name of pkl file containing welch values
filename = 'C:/Users/torterma/Documents/Projets_Osmose/Sciences/2_FirstSoundscapeAnalysis/E1_LTAS_all.npz'

npz_mat= np.load(filename, allow_pickle = True)
f=npz_mat['Freq']
Sxx=npz_mat['LTAS']
t=npz_mat['time']

fig, ax = plt.subplots(figsize=(40,15))
im=ax.imshow(Sxx[1:160], aspect='auto', extent=[t[0], t[-1], f[1], f[160]],origin = 'lower', vmin = 60, vmax = 80)

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


# Compute PSD

RMSlevel = 10 * np.log10(np.nanmean(10 ** (Sxx.transpose() / 10), axis=0))


percen = [1, 5, 50, 95, 99]
p = np.nanpercentile(Sxx.transpose(), percen, 0, interpolation='linear')


fig, ax = plt.subplots(figsize=(30,15))
ax.plot(f, RMSlevel, color='k', label='RMS level')


for i in range(len(p)):
    plt.plot(f, p[i, :], linewidth=2, label='%s %% percentil' % percen[i])
            










