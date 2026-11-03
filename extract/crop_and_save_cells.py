import skimage.io as skio
import numpy as np
import os
import seaborn as sns
import nd2

from lib.process_3d_cell import interpolate_z
from lib.utilities import list_to_array
from lib.preprocessing import correct_bleaching, check_mask
from lib.isolate_cell import crop_zstack

# =========================
# USER INPUT
# =========================

# Input files
image_path = "path_to_image_file.nd2"              # Supports .nd2 or .tif
mask_path = "path_to_mask_file.tif"
bleaching_mask_path = "path_to_bleaching_mask.tif"
cell_info_path = "path_to_cell_info.txt"           # Text file with cell IDs and time ranges

# Output settings
output_directory = "path_to_output_directory/"
dataset_label = "dataset_label"

# =========================
# IMAGE SCALE PARAMETERS
# =========================

# Confocal example values (adjust if needed)
scale = 63.69 / 512   # µm per pixel
z_width = 0.9         # µm per z-slice
frame_time = 20       # seconds per frame


# =========================
# PREPARE OUTPUT DIRECTORY
# =========================
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)
else:
    if len(os.listdir(output_directory)) != 0:
        user_in = input('Warning: output directory not empty. Do you want to continue? [y]/n')
        if not (user_in == 'y' or user_in == ""):
            raise OSError('Aborted by user')

# =========================
# LOAD CELL INFO
# =========================
# Expected format: columns [cell_id, start_frame, end_frame]
cell_info = np.loadtxt(cell_info_path, dtype=int, skiprows=1, usecols=(0, 1, 2))

# =========================
# PLOTTING STYLE
# =========================
sns.set_theme('paper')
sns.set_style('whitegrid')

# =========================
# LOAD MASKS
# =========================
mask = skio.imread(mask_path, plugin="tifffile")
bleaching_mask = skio.imread(bleaching_mask_path, plugin="tifffile")

# =========================
# VALIDATE MASK
# =========================
print('Verifying mask...', end='')
check_mask(mask, cell_info)
print('\rMask verification complete')

# =========================
# LOAD IMAGE DATA
# =========================
print('Reading image...', end='')

extension = image_path.split('.')[-1]

if extension == 'nd2':
    image = nd2.imread(image_path)
elif extension in ['tif', 'tiff']:
    image = skio.imread(image_path, plugin="tifffile")
else:
    raise RuntimeError("Image file is neither tif nor nd2.")

# If image has multiple channels, select fluorescence channel
if len(image.shape) == 5:
    image = image[:, :, 0, :, :]

print('\rImage reading complete')

# =========================
# BLEACHING CORRECTION
# =========================
print('Performing bleaching correction...', end='')

image = correct_bleaching(
    image,
    bleaching_mask,
    plot_fit=True,
    print_coeffs=False
)

print('\rBleaching correction complete')

# =========================
# PROCESS EACH CELL
# =========================
for line in range(len(cell_info)):

    cell_id = cell_info[line, 0]
    print('\r---- Working on cell', line + 1, '/', len(cell_info),
          '(id =', cell_id, ') ----')

    timerange = range(cell_info[line, 1], cell_info[line, 2])

    cell_timelapse = []

    # ---- LOOP OVER TIME ----
    for t in timerange:
        cell = crop_zstack(image[t], mask[t], cell_id)

        try:
            # Interpolate z-dimension to uniform scaling
            cell = interpolate_z(cell, scale, z_width)
        except IndexError:
            print('Mask error in frame', t + 1)

        cell_timelapse.append(cell)

    # Convert list to array
    cell_timelapse = list_to_array(cell_timelapse)

    # =========================
    # SAVE OUTPUT
    # =========================
    filename = os.path.join(
        output_directory,
        dataset_label + '_cropped_' + str(cell_id) + '_' + str(timerange[0]) + '.tif'
    )

    skio.imsave(
        filename,
        cell_timelapse,
        plugin="tifffile",
        imagej=True,
        resolution=(1 / scale, 1 / scale),
        metadata={
            'spacing': 1 / scale,
            'unit': 'um',
            'axes': 'TZYX',
            'fps': 1 / frame_time
        }
    )
