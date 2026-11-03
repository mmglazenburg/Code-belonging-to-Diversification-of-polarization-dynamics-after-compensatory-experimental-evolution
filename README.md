# Code belonging to Diversification of budding yeast polarization dynamics after compensatory experimental evolution
_M. M. Glazenburg & L. Laan._

This code belongs to the analysis performed in the manuscript titled Diversification of budding yeast polarization dynamics after compensatory experimental evolution.

## How to use - analyzing polarization events
This code can be used to repeat the analyses described in the main manuscript, i.e. segmenting polarization events, separating spots and and extracting observables. This requires 3D live cell fluorescence imaging data, a segmentation mask and an annotation of polarization events as input.

Sample data can be retrieved from our previous data repository found at https://doi.org/10.4121/90dc93d5-445f-463d-854e-70f43607ab01.

This code should be used as follows.

### (1) Extract events and observables
Individual polarization events need to be extracted from the main field of view and saved separately. Then, these cropped cells are used to calculate polarization characteristics. This is done by running the scripts in the `extract` folder in this order:

1. Use `crop_and_save_cells.py` to perform bleaching correction and extract single cells/polarization events from raw imaging data.
    - This takes as input:
        1. 3D fluorescent image timelapse as one .tif or .nd2 file
        2. Corresponding 2D cell mask where each cell has a unique and persistant integer identifier
        3. `cell_info.txt` with annotation info containing the cell ID, starting and ending timepoint, and cell type (mother [m] or daughter [d])
    - This outputs:
        1. 3D timelapses of single cells/polarization events, bleaching corrected and cropped from the field of view
2. Use `extract_multispot_masks.py` to create .tif files for each polarization event in which each spot has a unique identifier
   - This takes as input:
        1. A directory of segmented singe cells/polarization events as outputted by `crop_and_save_cells.py`
   - This outputs:
        1. 3D timelapse masks corresponding to each polarization event, in which spots identified as being separate are marked with a unique integer (background is 0), saved in a directory named `cluster_masks` inside the directory of the cells
3. Use `extract_properties_main.py ` to create .txt files containing all observables
    - This takes as input:
        1. A directory of segmented single cells/polarization events as outputted by `crop_and_save_cells.py`
        2. A directory of corresponding multispot masks as outputted by `extract_multispot_masks.py`, named `cluster_masks` and located inside the directory of the cells (if `use_masks` is set to `True`)
    - This outputs:
        1. .txt files containing calculated observables for the 'winning'/final spot of each polarization event

### (2) Plot and analyse observables
After obtaining the .txt files containing calculated observables, they can be plotted and statistically analysed using the scripts in the `analyse/property_comparisons` folder.

1. Use the other files to create the plots to compare observables.
    - These takes as input:
       1. Directories in which the .txt files outputted by `extract_properties_main.py ` are stored
   - These output, depending on the script:
     1. `individual_properties_compared.py`: scatter + violin plots of individual polarization characteristics with statistical analysis (toggle the desired observable to plot by uncommenting)
     2. `compare_mean_intensities.py`: mean intensity traces + individual traces
     3. `compare_mobility.py`: spot mobility during establishment and maintenance with statistical analysis
     4. `compare_variance.py`: fold changes of distribution variances for all observables compared to some reference
     5. `matrix_visualisation.py`: fold changes of distribution means for all observables compared to some reference
     6. `all_properties.py`: plots all properties + intensity traces of one condition together
2. PCA analysis is found under `analyse/pca/pca.py`
    - This takes as input:
     1. Directories in which the .txt files outputted by `extract_properties_main.py ` are stored
    - This outputs:
      1. PCA analysis and plots as appearing in the manuscript

### (3) (optional) Assess the effect of thresholding
Use the scripts in the subfolder `analyse/assess_thresholding` to see the effect of applying different thresholds and blurring to segmentation of the polarized signal.

1. `histograms.py` plots histograms of a single cell in unpolarized and polarized state, together with images, indicating where various threshold levels fall. It calculates thresholded voxel fractions in each state for varying threshold levels and saves these to .txt files.
2. `pixel_fractions.py` takes the .txt files outputted by `histograms.py` and plots them as a function of threshold level.
