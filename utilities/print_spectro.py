import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import os
from matplotlib import colormaps
from PIL import Image, ImageOps, ImageFilter
import glob

os.chdir(r'U:/Documents_U/Git')

# Load the WAV file
# path = 'Y:/Bioacoustique/APOCADO2/Campagne 6/PASSE PARTOUT/bouts rouges/7178/analysis/C6D3/results/3_96000 wav'
path = r'C:\Users\dupontma2\Downloads'
sample_rate, audio_data = wavfile.read(os.path.join(path, '2022_05_29T22_42_36.wav'))



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
plt.xticks([], [])
plt.yticks([], [])
plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # delete white borders

plt.show()

Nbech = len(audio_data)
size_x = (Nbech - window_size) / overlap_samples
size_y = nfft / 2

print(f'\nX: {size_x}\nY: {size_y}')

#%%
base_folder = r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\Documents\spectro exemples'
name_folder = r'test_quality_1'
name_image = next(iter(glob.glob(os.path.join(base_folder, name_folder, '*.png'))))
original_image = Image.open(name_image)
gray_image = Image.open(name_image).convert('L')

# plt.imshow(original_image)
# plt.axis('off')  # Optional: Turn off axes

# # original
# out_path = os.path.join(os.path.dirname(name_image), os.path.basename(name_image).split('.')[0] + f'_same.png')
# original_image.save(out_path)

# grayscale
out_path = os.path.join(os.path.dirname(name_image), os.path.basename(name_image).split('.')[0] + f'_grey.png')
gray_image.save(out_path)

# grayscale
quality_factor = 90
out_path = os.path.join(os.path.dirname(name_image), os.path.basename(name_image).split('.')[0] + f'_grey_quality{quality_factor}.jpg')
gray_image.save(out_path, quality=quality_factor)

# # reconstructed rgb
# gray_array = np.array(gray_image)
# cmap = plt.cm.viridis
# norm_gray_array = gray_array / 255.0
# viridis_array = cmap(norm_gray_array)
# viridis_array_uint8 = (viridis_array[:, :, :3] * 255).astype(np.uint8)
# viridis_image = Image.fromarray(viridis_array_uint8)
# out_path = os.path.join(os.path.dirname(name_image), os.path.basename(name_image).split('.')[0] + f'_reconstructed.png')
# viridis_image.save(out_path)

# # auto enhanced
# enhanced_gray_image = ImageOps.autocontrast(gray_image)
# enhanced_gray_array = np.array(enhanced_gray_image)
# cmap = plt.cm.viridis
# norm_gray_array = enhanced_gray_array / 255.0
# viridis_array = cmap(norm_gray_array)
# viridis_array_uint8 = (viridis_array[:, :, :3] * 255).astype(np.uint8)
# viridis_image = Image.fromarray(viridis_array_uint8)
# out_path = os.path.join(os.path.dirname(name_image), os.path.basename(name_image).split('.')[0] + f'_reconstructed_enhanched.png')
# viridis_image.save(out_path)

# # Manually adjust contrast
# for c in [1.2, 1.8]:
    
#     gray_array = np.array(gray_image, dtype=np.float32)
#     adjusted_gray_array = 128 + (gray_array - 128) * c
#     adjusted_gray_array = np.clip(adjusted_gray_array, 0, 255)
#     adjusted_gray_image = Image.fromarray(adjusted_gray_array.astype(np.uint8))
#     cmap = plt.cm.viridis
#     norm_adjusted_gray_array = adjusted_gray_array / 255.0
#     viridis_array = cmap(norm_adjusted_gray_array)
#     viridis_array_uint8 = (viridis_array[:, :, :3] * 255).astype(np.uint8)
#     viridis_image = Image.fromarray(viridis_array_uint8)
#     viridis_image
    
#     out_path = os.path.join(os.path.dirname(name_image), os.path.basename(name_image).split('.')[0] + f'_reconstructed_manual_contrast_c{c}.png')
#     viridis_image.save(out_path)

# # clarity test
# for c in range(1, 110, 20):
#     enhanced_gray_image2  = gray_image.filter(ImageFilter.UnsharpMask(radius=2, percent=c))
#     enhanced_gray_array2 = np.array(enhanced_gray_image2)
#     cmap = plt.cm.viridis
#     norm_gray_array2 = enhanced_gray_array2 / 255.0
#     viridis_array = cmap(norm_gray_array2)
#     viridis_array_uint8 = (viridis_array[:, :, :3] * 255).astype(np.uint8)
#     viridis_image = Image.fromarray(viridis_array_uint8)
#     out_path = os.path.join(os.path.dirname(name_image), os.path.basename(name_image).split('.')[0] + f'_reconstructed_clarity{c}.png')
#     viridis_image.save(out_path)

# contrast + clarity
name_image = next(iter(glob.glob(os.path.join(base_folder, name_folder, '*.jpg'))))

clarity_factor = 20
contrast_factor = 1.4
enhanced_gray_image3  = gray_image.filter(ImageFilter.UnsharpMask(radius=2, percent=clarity_factor))
gray_array = np.array(enhanced_gray_image3, dtype=np.float32)
adjusted_gray_array = 128 + (gray_array - 128) * contrast_factor
adjusted_gray_array = np.clip(adjusted_gray_array, 0, 255)
adjusted_gray_image = Image.fromarray(adjusted_gray_array.astype(np.uint8))
cmap = plt.cm.viridis
norm_adjusted_gray_array = adjusted_gray_array / 255.0
viridis_array = cmap(norm_adjusted_gray_array)
viridis_array_uint8 = (viridis_array[:, :, :3] * 255).astype(np.uint8)
viridis_image = Image.fromarray(viridis_array_uint8)
out_path = os.path.join(os.path.dirname(name_image), os.path.basename(name_image).split('.')[0] + f'_reconstructed_clarity{clarity_factor}_contrast{contrast_factor}.png')
viridis_image.save(out_path)



#%%

# Load the grayscale spectrogram image
gray_image = Image.open(r'L:\\acoustock\\Bioacoustique\\DATASETS\\APOCADO\\Documents\\spectro exemples\\test_quality_1\\2022_07_06T23_59_47_1_0.png').convert('L')

# Convert grayscale image to numpy array
gray_array = np.array(gray_image)


# Apply Viridis colormap
cmap = colormaps['grey']
viridis_image = cmap(gray_array)

# Convert to uint8 (0-255) and remove alpha channel
viridis_image = (viridis_image[:, :, :3] * 255).astype(np.uint8)

# Plot the resulting RGB image
plt.imshow(viridis_image)
plt.axis('off')  # Optional: Turn off axes
plt.show()

# Save the resulting RGB image
viridis_image = Image.fromarray(viridis_image)
viridis_image.save(r'L:\acoustock\Bioacoustique\DATASETS\APOCADO\Documents\spectro exemples\2022_07_06T23_59_47_1_0_reconstructed.png')





























