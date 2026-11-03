import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Ensure text remains editable in exported SVGs
plt.rcParams['svg.fonttype'] = 'none'

# =========================
# USER INPUT
# =========================
# Directory containing output .txt files from previous analysis
directory = "path_to_your_results_directory/"

# =========================
# LOAD DATA
# =========================
false_pixels = np.loadtxt(os.path.join(directory, 'false_pixels.txt'))
false_pixels_blurred = np.loadtxt(os.path.join(directory, 'false_pixels_blurred.txt'))
polarized_pixels = np.loadtxt(os.path.join(directory, 'polarized_pixels.txt'))
polarized_pixels_blurred = np.loadtxt(os.path.join(directory, 'polarized_pixels_blurred.txt'))

# Standard deviation values used for thresholding
stds = [1, 2, 3, 4, 5, 6, 7]

# =========================
# PLOTTING STYLE SETTINGS
# =========================
sns.set_theme('paper')
sns.set_style('whitegrid')

# =========================
# PLOTTING
# =========================
plt.figure(figsize=(3, 6.2))

# ---- UNPOLARIZED STATE ----
plt.subplot(2, 1, 1)

plt.errorbar(
    stds,
    np.nanmean(false_pixels, axis=0),
    yerr=np.nanstd(false_pixels, axis=0),
    label='Unsmoothed masking',
    fmt='-',
    marker='o',
    capsize=5
)

plt.errorbar(
    stds,
    np.nanmean(false_pixels_blurred, axis=0),
    yerr=np.nanstd(false_pixels_blurred, axis=0),
    label='Smoothed masking',
    fmt='--',
    marker="^",
    capsize=5
)

plt.title('Unpolarized state')
plt.xlabel(r'Number of standard deviations $k$')
plt.ylabel('Thresholded voxel fraction [%]')
plt.xticks(np.arange(min(stds), max(stds) + 1, 1.0))
plt.legend()

# ---- POLARIZED STATE ----
plt.subplot(2, 1, 2)

plt.errorbar(
    stds,
    np.nanmean(polarized_pixels, axis=0),
    yerr=np.nanstd(polarized_pixels, axis=0),
    label='Unsmoothed masking',
    fmt='-',
    marker='o',
    capsize=5
)

plt.errorbar(
    stds,
    np.nanmean(polarized_pixels_blurred, axis=0),
    yerr=np.nanstd(polarized_pixels_blurred, axis=0),
    label='Smoothed masking',
    fmt='--',
    marker="^",
    capsize=5
)

plt.title('Polarized state')
plt.xlabel(r'Number of standard deviations $k$')
plt.ylabel('Thresholded voxel fraction [%]')
plt.xticks(np.arange(min(stds), max(stds) + 1, 1.0))
plt.legend()

plt.tight_layout()
plt.show()