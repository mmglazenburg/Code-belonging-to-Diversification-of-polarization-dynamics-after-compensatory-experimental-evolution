# Code belonging to Quantitative Characterization of Budding Yeast Polarization at the Mesoscale
_M. M. Glazenburg, E. Geurken & L. Laan._

This repository contains the code underlying the publication _Quantitative Characterization of Budding Yeast Polarization at the Mesoscale_ (https://doi.org/10.1103/gp4c-y48t).

_AI disclaimer: generative AI was jused for the purpose of preparing the code for publication in a public repository, as well as for optimizing very specific functionalities. The majority of the code was conceived without AI assistance._

## How to use
This code can be used to repeat the analyses described in the main publication, i.e. segmenting polarization events and extracting observables. This requires 3D live cell fluorescence imaging data, a segmentation mask and an annotation of polarization events as input.

Sample data can be retrieved from the corresponding data repository found at https://doi.org/10.4121/90dc93d5-445f-463d-854e-70f43607ab01.

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
2. Use `extract_properties_main.py ` to create .txt files containing all observables
    - This takes as input:
        1. A directory of segmented singe cells/polarization events as outputted by `crop_and_save_cells.py`
    - This outputs:
        1. .txt files containing calculated observables

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

### (3) (optional) Assess the effect of thresholding
Use the scripts in the subfolder `analyse/assess_thresholding` to see the effect of applying different thresholds and blurring to segmentation of the polarized signal.

1. `histograms.py` plots histograms of a single cell in unpolarized and polarized state, together with images, indicating where various threshold levels fall. It calculates thresholded voxel fractions in each state for varying threshold levels and saves these to .txt files.
2. `pixel_fractions.py` takes the .txt files outputted by `histograms.py` and plots them as a function of threshold level.
