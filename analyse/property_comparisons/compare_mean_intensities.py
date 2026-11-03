import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import seaborn as sns
import scienceplots

from lib.utilities import get_signal_start_end

# Ensure text remains editable in SVG export
plt.rcParams['svg.fonttype'] = 'none'

# =========================
# SELECT DATASETS
# =========================
# Choose ONE configuration by uncommenting it

# Example 1: Comparison with/without denoising
# directories = ["path_to_dataset_1/", "path_to_dataset_2/"]
# labels = ['With Denoise.ai', r'Without Denoise.ai']
# palette = sns.color_palette('Paired')
# align_onset = False

# Example 2: Mutant comparison
# directories = ["path_to_dataset_1/", "path_to_dataset_2/", "path_to_dataset_3/"]
# labels = ['WT', r'rsr1Δ', r'axl2Δ rax1Δ']
# palette = sns.color_palette('Dark2')
# align_onset = False

# Example 3: Different proteins
# directories = ["path_to_dataset_1/", "path_to_dataset_2/", "path_to_dataset_3/"]
# labels = ['Cdc42', 'Gic2PBD', 'Spa2']
# palette = sns.color_palette()
# align_onset = False

# Example 4: Single dataset
# directories = ["path_to_dataset/"]
# labels = ['Dataset']
# palette = ['k']
# align_onset = False

# ---- ACTIVE CONFIGURATION ----
directories = [
    "path_to_dataset_1/",
    "path_to_dataset_2/",
    "path_to_dataset_3/",
    "path_to_dataset_4/",
    "path_to_dataset_5/"
]

labels = [
    'Condition 1',
    'Condition 2',
    'Condition 3',
    'Condition 4',
    'Condition 5'
]

palette = ['k', *sns.color_palette()]
align_onset = False

# =========================
# PARAMETERS
# =========================
scale = 63.69 / 512   # µm per pixel
frame_time = 20       # seconds per frame

# Line styles for mean curves
dashes = [
    (1, 0), (3, 1), (1, 1, 3, 1),
    (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)
]

# =========================
# LOAD DATA
# =========================
all_intensity_blocks = []
max_len = 0

for directory in directories:
    intensities_aligned = np.loadtxt(directory + 'all_intensity_traces.txt')
    all_intensity_blocks.append(intensities_aligned)

    if intensities_aligned.shape[1] > max_len:
        max_len = intensities_aligned.shape[1]

# =========================
# PLOTTING SETUP
# =========================
sns.set_theme('notebook')
sns.set_style('ticks')

plt.figure(figsize=(5, 4))

# Time axis (in minutes)
time = np.arange(max_len) * frame_time / 60
time = time[::-1]   # Time counts down to bud emergence at 0

# =========================
# ALIGNMENT OPTION
# =========================
if align_onset:
    # ---- ALIGN BY POLARITY ONSET ----

    pstart_max = 0

    # Find latest onset among datasets
    for intensity_block in all_intensity_blocks:
        mean = np.nanmean(intensity_block, axis=0)
        pstart, pend = get_signal_start_end(mean)
        if pstart > pstart_max:
            pstart_max = pstart

    intensity_front_padded = []

    # Align all datasets to this onset
    for intensity_block in all_intensity_blocks:
        mean = np.nanmean(intensity_block, axis=0)
        pstart, pend = get_signal_start_end(mean)

        intensity_block_front_padded = np.pad(
            intensity_block,
            (pstart_max - pstart, 0),
            'constant',
            constant_values=(np.nan, np.nan)
        )

        intensity_front_padded.append(intensity_block_front_padded)

    # Recompute time axis
    max_len = max([block.shape[1] for block in intensity_front_padded])
    min_len = min([block.shape[1] for block in intensity_front_padded])

    time = np.arange(max_len) * frame_time / 60
    time = time[::-1]
    time -= (max_len - min_len) * frame_time / 60

    # Plot
    for i, intensity_block in enumerate(intensity_front_padded):
        padded = np.pad(
            intensity_block,
            (0, max_len - intensity_block.shape[1]),
            'constant',
            constant_values=(np.nan, np.nan)
        )

        for intensity in padded:
            plt.plot(time, intensity, alpha=0.1, c=palette[i], linewidth=1)

        plt.plot(time, np.nanmean(padded, axis=0),
                 label=labels[i], c=palette[i],
                 lw=1.7, dashes=dashes[i])

else:
    # ---- ALIGN BY END (DEFAULT) ----
    for i, intensity_block in enumerate(all_intensity_blocks):
        padded = np.pad(
            intensity_block,
            (max_len - intensity_block.shape[1], 0),
            'constant',
            constant_values=(np.nan, np.nan)
        )

        for intensity in padded:
            plt.plot(time, intensity, alpha=0.1, c=palette[i], linewidth=1)

        plt.plot(time, np.nanmean(padded, axis=0),
                 label=labels[i], c=palette[i],
                 lw=1.7, dashes=dashes[i])

# =========================
# FORMATTING
# =========================
plt.xlabel('Time until bud emergence (min)')
plt.gca().xaxis.set_inverted(True)  # End at 0 (bud emergence)
plt.ylabel('Normalized intensity')
plt.ticklabel_format(axis='y', style='sci', scilimits=[-2, 2])

plt.legend()
plt.tight_layout()
plt.show()