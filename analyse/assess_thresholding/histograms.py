import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.filters
import seaborn as sns

# Ensure text remains editable in exported SVGs
plt.rcParams['svg.fonttype'] = 'none'

from lib.process_3d_cell import variance_threshold, normalize_image
from lib.utilities import get_signal_start_end

# =========================
# USER INPUT
# =========================
# Set your data directory here (should contain .tif files)
directory = "path_to_your_data_directory/"

# =========================
# PLOTTING STYLE SETTINGS
# =========================
sns.set_theme('paper')
sns.set_style('whitegrid')

# =========================
# LOAD FILES
# =========================
image_files = os.listdir(directory)

# Standard deviations used for thresholding
stds = [1, 2, 3, 4, 5, 6, 7]

# Arrays to store results
false_pixels = np.zeros((len(image_files), len(stds)))
false_pixels_blurred = np.zeros((len(image_files), len(stds)))
polarized_pixels = np.zeros((len(image_files), len(stds)))
polarized_pixels_blurred = np.zeros((len(image_files), len(stds)))

# =========================
# MAIN LOOP OVER IMAGES
# =========================
for i, file in enumerate(image_files):

    # Process only TIFF files
    if file.split('.')[-1] == 'tif':
        print(file)

        image_path = os.path.join(directory, file)

        # ---- LOAD AND NORMALIZE IMAGE ----
        image = skio.imread(image_path, plugin="tifffile")
        image = normalize_image(image)

        # ---- COMPUTE THRESHOLDS ----
        thresholds = [variance_threshold(image, std_no=std) for std in stds]
        colors = ['#292f56', '#09507f', '#007498', '#0097a3',
                  '#00bca1', '#43df8b', '#acfa70']

        # Use std=3 as reference threshold for polarization window
        standard_threshold = thresholds[2]
        pstart, pend = get_signal_start_end(image, signal_threshold=standard_threshold)

        # ---- DEFINE REGIONS ----
        image_peak = image[pend]        # polarized region
        image_start = image[0:5]        # unpolarized region

        image_max = np.max(image_peak, axis=0)

        # =========================
        # PLOTTING
        # =========================
        plt.figure(figsize=(4.5, 7))

        # ---- HISTOGRAM: RAW IMAGE ----
        plt.subplot(3, 1, 1)
        plt.hist(image_start[image_start > 0], bins=40, density=True,
                 color='darkred', alpha=0.6, label='unpolarized', lw=1, edgecolor='k')
        plt.hist(image_peak[image_peak > 0], bins=50, density=True,
                 color='darkred', alpha=0.3, label='polarized', lw=1, edgecolor='k')

        plt.xlabel('Normalized voxel intensity')
        plt.ylabel('log(density)')
        plt.yscale('log')

        # Plot thresholds
        for j, threshold in enumerate(thresholds):
            plt.axvline(threshold, c=colors[j], ls='--')

        plt.legend()
        xlim = plt.gca().get_xlim()

        # ---- APPLY GAUSSIAN SMOOTHING ----
        image_blurred = skimage.filters.gaussian(image, sigma=(0, 1, 1, 1))
        image_blurred_start = image_blurred[0:5]
        image_blurred_peak = image_blurred[pend]

        # ---- HISTOGRAM: BLURRED IMAGE ----
        plt.subplot(3, 1, 2)
        plt.hist(image_blurred_start[image_blurred_start > 0], bins=40, density=True,
                 color='darkred', alpha=0.6, label='unpolarized', lw=1, edgecolor='k')
        plt.hist(image_blurred_peak[image_blurred_peak > 0], bins=50, density=True,
                 color='darkred', alpha=0.3, label='polarized', lw=1, edgecolor='k')

        plt.xlabel('Normalized voxel intensity')
        plt.ylabel('log(density)')
        plt.xlim(xlim)
        plt.yscale('log')

        for j, threshold in enumerate(thresholds):
            plt.axvline(threshold, c=colors[j], ls='--')

        plt.legend()

        # ---- MASK VISUALIZATION: RAW ----
        plt.subplot(3, 2, 5)
        plt.title('Unsmoothed masking')
        plt.imshow(image_max, cmap='binary')
        plt.contour(image_max, thresholds, colors=colors)
        plt.xticks([])
        plt.yticks([])

        # ---- MASK VISUALIZATION: BLURRED ----
        plt.subplot(3, 2, 6)
        plt.title('Smoothed masking')
        image_peak_blurred = image_blurred[pend]
        image_max_blurred = np.max(image_peak_blurred, axis=0)
        plt.imshow(image_max, cmap='binary')
        plt.contour(image_max_blurred, thresholds, colors=colors)
        plt.xticks([])
        plt.yticks([])

        # =========================
        # METRICS COMPUTATION
        # =========================
        for j, threshold in enumerate(thresholds):

            # Gaussian smoothing for initial region
            sigma = (0, 1, 1, 1)
            image_start_blurred = skimage.filters.gaussian(image_start, sigma=sigma)

            # % false positives (unpolarized region classified as signal)
            false_pixels_blurred[i, j] = (
                np.sum(image_start_blurred > threshold) /
                image_start_blurred.size * 100
            )

            false_pixels[i, j] = (
                np.sum(image_start > threshold) /
                image_start.size * 100
            )

            # % detected polarized pixels
            image_peak_blurred = skimage.filters.gaussian(image_peak, sigma=1)

            polarized_pixels[i, j] = (
                np.sum(image_peak > threshold) /
                image_peak.size * 100
            )

            polarized_pixels_blurred[i, j] = (
                np.sum(image_peak_blurred > threshold) /
                image_peak_blurred.size * 100
            )

        plt.tight_layout()
        plt.close()

    else:
        # Fill with NaNs if file is not a TIFF image
        false_pixels[i, :] = np.full(len(stds), np.nan)
        false_pixels_blurred[i, :] = np.full(len(stds), np.nan)
        polarized_pixels[i, :] = np.full(len(stds), np.nan)
        polarized_pixels_blurred[i, :] = np.full(len(stds), np.nan)

# =========================
# SAVE RESULTS
# =========================
np.savetxt(os.path.join(directory, 'false_pixels.txt'), false_pixels)
np.savetxt(os.path.join(directory, 'false_pixels_blurred.txt'), false_pixels_blurred)
np.savetxt(os.path.join(directory, 'polarized_pixels.txt'), polarized_pixels)
np.savetxt(os.path.join(directory, 'polarized_pixels_blurred.txt'), polarized_pixels_blurred)

plt.close('all')