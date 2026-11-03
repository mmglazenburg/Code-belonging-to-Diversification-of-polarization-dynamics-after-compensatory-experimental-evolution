import skimage.io as skio
import os

from lib.process_3d_cell import normalize_image, variance_threshold, mask_polarity
from lib.multispot_handling import make_clustered_mask


# =========================
# SELECT DATASET
# =========================
# Choose ONE dataset by setting the directory

# Example 1
# directory = "path_to_dataset_1/cells/"

# Example 2
# directory = "path_to_dataset_2/cells/"

# Example 3 (test data)
# directory = "path_to_test_dataset/cells/"

# ---- ACTIVE CONFIGURATION ----
directory = "path_to_dataset/cells/"


# =========================
# PARAMETERS
# =========================
std_no = 4      # number of standard deviations the threshold is defined away from the signal mean
plot = True     # set True if you want to see the clustering results for each cell

scale = 63.69/512 # um/pixel
frame_time = 20 # seconds per frame


# =========================
# RUN MASK EXTRACTION
# =========================
image_files = os.listdir(directory)

for i, file in enumerate(image_files):
    print(f'\rProcessing file {i+1}/{len(image_files)} {file}', end='')
    if file.split('.')[-1] == 'tif':

        image_path = directory + file

        # READ IMAGE AND NORMALIZE TO AVERAGE CYTOSOL INTENSITY AT THE START
        image = skio.imread(image_path, plugin="tifffile")
        image = normalize_image(image)

        # DETERMINE THRESHOLD AND MASK IMAGE
        threshold = variance_threshold(image, std_no=std_no)

        masked_image = mask_polarity(image, threshold)

        clustered_labels = make_clustered_mask(masked_image, plot=plot)

        filename = directory + 'cluster_masks/' + file.split('.')[0] + '_cluster_mask.tif'
        skio.imsave(filename, clustered_labels, plugin="tifffile", imagej=True, metadata={'axes': 'TZYX', 'threshold_n_sigma': str(std_no)}, check_contrast=False)


# SAVE TXT FILE WITH THE USED THRESHOLD VALUE IN THE SAME FOLDER
with open(directory + 'cluster_masks/thr_info.txt', 'w') as file:
    file.write(f"Threshold used: {std_no} * sigma")
