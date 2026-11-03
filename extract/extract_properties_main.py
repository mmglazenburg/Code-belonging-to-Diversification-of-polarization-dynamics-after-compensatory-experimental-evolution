from lib.prepare_properties import extract_all_properties

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

directory = "C:/Data/cropped_cells/"

# =========================
# PARAMETERS
# =========================
scale = 63.69 / 512   # µm per pixel
frame_time = 20       # seconds per frame

# =========================
# RUN PROPERTY EXTRACTION
# =========================
print('Working on directory ' + directory)


extract_all_properties(
    directory,
    plot_ind_traces=True
)