import os
import numpy as np
import pandas as pd
import skimage.io as skio
import matplotlib.pyplot as plt
from skimage import measure
import scipy as sc
from pathlib import Path
import math

from lib.process_3d_cell import variance_threshold, normalize_image, mask_polarity
from lib.utilities import align_traces, get_signal_start_end


def extract_all_properties(directory,
                           threshold_stds=3,
                           plot_ind_traces=False):
    """
    Extracts polarity-related properties from 3D cell timelapse data and saves
    them to text files in the parent directory.

    Parameters
    ----------
    directory : str
        Directory containing cropped cell timelapse .tif files.
    threshold_stds : int
        Threshold level for segmentation (in standard deviations).
    plot_ind_traces : bool
        Plot intensity traces per cell.
    """

    image_files = os.listdir(directory)

    intensity_traces = []
    pstarts = []
    volume_traces = []
    window_lengths = []
    peak_intensities = []
    peak_volumes = []
    circularities = []
    all_step_sizes = []
    cytosol_intensities = []
    cell_sizes = []
    replicates = []

    max_array_length = 0

    i = 0
    for file in image_files:
        print(f'\rProcessing file {i+1}/{len(image_files)} {file}', end='')

        if file.split('.')[-1] != 'tif':
            continue

        i += 1
        image_path = directory + file

        # LOAD IMAGE
        image_raw = skio.imread(image_path, plugin="tifffile")
        image = normalize_image(image_raw)

        # SEGMENTATION
        threshold = variance_threshold(image, std_no=threshold_stds)
        image_masked = mask_polarity(image, threshold)

        # INTENSITY TRACE
        intensity = np.sum(image_masked, axis=(1, 2, 3))
        intensity_traces.append(intensity)

        intensity_filtered = sc.signal.savgol_filter(intensity, max(int(len(intensity) / 5), 4), 3)
        pstart, pend = get_signal_start_end(intensity_filtered)

        pstarts.append(pstart)
        window_lengths.append(pend - pstart)

        peak_min = max([pend - 2, pstart])
        peak_max = min([pend + 3, len(image)])

        # OPTIONAL TRACE PLOTS
        if plot_ind_traces:
            nrows, ncols = 4, 5
            ax = plt.subplot(nrows, ncols, (i - 1) % (nrows * ncols) + 1)

            time = np.linspace(0, len(intensity_filtered) / 3, len(intensity_filtered))
            ax.axvspan(pstart / 3, pend / 3, alpha=0.2, color='darkred')

            ax.scatter(time, intensity, c='k', s=3)
            ax.plot(time, intensity_filtered)

            if i % (nrows * ncols) == 0:
                plt.show()

        # PEAK INTENSITY & VOLUME
        peak_intensities.append(np.mean(intensity[peak_min:peak_max]))

        volumes = np.array([len(frame[frame > 0]) for frame in image_masked], dtype=float)
        volume_traces.append(volumes)
        peak_volumes.append(np.mean(volumes[peak_min:peak_max]))

        # CYTOSOL INTENSITY
        image_start_raw = np.max(image_raw[0:5], axis=1)
        cytosol_intensities.append(np.mean(image_start_raw[image_start_raw > 0]))

        # STEP SIZE
        image_masked_unsmoothed = image * (image > threshold)
        image_masked_polarized = image_masked_unsmoothed[pstart:]

        coms_polarized = np.array([
            sc.ndimage.center_of_mass(frame)
            for frame in image_masked_polarized
        ])

        mask_slice = image_raw.copy()
        mask_slice[mask_slice > 0] = 1

        coms_mask = np.array([
            sc.ndimage.center_of_mass(frame)
            for frame in mask_slice
        ])

        coms_relative = coms_polarized - coms_mask[pstart:]

        step_sizes_xyz = np.diff(coms_relative, axis=0)
        step_sizes = np.sqrt(np.sum(step_sizes_xyz ** 2, axis=1))

        all_step_sizes.append(step_sizes)
        max_array_length = max(max_array_length, len(step_sizes))

        # CIRCULARITY
        image_masked_peak = image_masked[peak_min:peak_max]
        coms_peak = coms_polarized[peak_min - pstart:peak_max - pstart]
        com_cell = [image.shape[1] / 2, image.shape[2] / 2, image.shape[3] / 2]

        cell_circularities = np.zeros(len(image_masked_peak))

        for j, frame in enumerate(image_masked_peak):

            z, y, x = np.array(coms_peak[j]) - np.array(com_cell)

            if not np.isfinite(z):
                cell_circularities[j] = np.nan
                continue

            azimuth = np.degrees(np.arctan2(x, y))
            polar = np.degrees(np.arccos(z / np.sqrt(x**2 + y**2 + z**2)))

            image_xyrot = sc.ndimage.rotate(frame, -azimuth, axes=(1, 2))
            image_zrot = sc.ndimage.rotate(image_xyrot, -polar, axes=(0, 1))
            projection = np.max(image_zrot, axis=0)

            contours = measure.find_contours(projection, 1)

            for contour_candidate in contours:
                y_max, x_max = np.unravel_index(np.argmax(projection), projection.shape)

                X = contour_candidate[:, 1]
                Y = contour_candidate[:, 0]

                if np.min(X) <= x_max <= np.max(X) and np.min(Y) <= y_max <= np.max(Y):
                    contour = contour_candidate

            try:
                x = contour[:, 1]
                y = contour[:, 0]

                area = 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
                area = abs(area)

                perim = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

                cell_circularities[j] = (4 * np.pi * area) / perim**2

            except UnboundLocalError:
                cell_circularities[j] = np.nan

        circularities.append(np.nanmean(cell_circularities))

        # CELL SIZE
        single_slice = image_raw[0, 0]
        cell_sizes.append(len(single_slice[single_slice > 0]))

        replicates.append(file.split('_')[0])

    if plot_ind_traces:
        plt.show()

    # ALIGN TRACES
    intensities_aligned = align_traces(intensity_traces, pstarts=pstarts)
    volumes_aligned = align_traces(volume_traces, pstarts=pstarts)

    all_step_sizes = [
        np.pad(arr, (0, max_array_length - len(arr)), constant_values=np.nan)
        for arr in all_step_sizes
    ]

    # SAVE RESULTS
    output_directory = Path(directory).parent

    np.savetxt(output_directory / 'window_lengths.txt',
               np.vstack((window_lengths, replicates)).T, fmt="%s")

    np.savetxt(output_directory / 'peak_intensities.txt',
               np.vstack((peak_intensities, replicates)).T, fmt="%s")

    np.savetxt(output_directory / 'peak_volumes.txt',
               np.vstack((peak_volumes, replicates)).T, fmt="%s")

    np.savetxt(output_directory / 'circularities.txt',
               np.vstack((circularities, replicates)).T, fmt="%s")

    np.savetxt(output_directory / 'cytosol_intensities.txt',
               np.vstack((cytosol_intensities, replicates)).T, fmt="%s")

    np.savetxt(output_directory / 'all_intensity_traces.txt', intensities_aligned)
    np.savetxt(output_directory / 'all_volume_traces.txt', volumes_aligned)

    np.savetxt(output_directory / 'step_sizes.txt',
               np.hstack((all_step_sizes, np.array(replicates).reshape(-1, 1))),
               fmt="%s")

    np.savetxt(output_directory / 'cell_sizes.txt',
               np.vstack((cell_sizes, replicates)).T, fmt="%s")


def split_step_sizes(all_step_sizes, window_lengths):
    """Split step sizes into establishment and maintenance phases."""

    step_size_during = np.zeros(len(all_step_sizes))
    step_size_after = np.zeros(len(all_step_sizes))
    time_to_budding = np.zeros(len(all_step_sizes))

    for i in range(len(all_step_sizes)):
        step_sizes = all_step_sizes[i]
        step_sizes = step_sizes[~np.isnan(step_sizes)]

        during = step_sizes[:int(window_lengths[i])]
        after = step_sizes[int(window_lengths[i]):]

        step_size_during[i] = np.nanmean(during)
        step_size_after[i] = np.nanmean(after) if len(after) >= 5 else np.nan
        time_to_budding[i] = len(after) + 1

    return step_size_during, step_size_after, time_to_budding


def create_property_dataframe(directories, dataset_labels,
                              frame_time=20, scale=63.69/512):
    """Combine all extracted properties into a single DataFrame."""

    window_lengths, peak_intensities, peak_volumes = [], [], []
    circularities, increase_rates, spot_densities = [], [], []
    step_sizes_during, step_sizes_after, times_to_budding = [], [], []
    cytosol_intensities, cell_sizes = [], []
    replicates, labels = [], []

    for i, directory in enumerate(directories):
        print('-- Working on directory ' + dataset_labels[i] + ' --')

        wl = np.loadtxt(directory + 'window_lengths.txt', usecols=0)
        pi = np.loadtxt(directory + 'peak_intensities.txt', usecols=0)
        pv = np.loadtxt(directory + 'peak_volumes.txt', usecols=0)
        circ = np.loadtxt(directory + 'circularities.txt', usecols=0)

        window_lengths.append(wl)
        peak_intensities.append(pi)
        peak_volumes.append(pv)
        circularities.append(circ)

        increase_rates.append(pi / wl)
        spot_densities.append(pi / pv)

        steps = np.loadtxt(directory + 'step_sizes.txt', dtype=str)[:, :-1].astype(float)
        d, a, t = split_step_sizes(steps, wl)

        step_sizes_during.append(d)
        step_sizes_after.append(a)
        times_to_budding.append(t)

        cytosol_intensities.append(np.loadtxt(directory + 'cytosol_intensities.txt', usecols=0))
        cell_sizes.append(np.loadtxt(directory + 'cell_sizes.txt', usecols=0))

        replicates.append(np.loadtxt(directory + 'window_lengths.txt', usecols=1, dtype=str))
        labels.append(np.array([dataset_labels[i]] * len(wl)))

    df = pd.DataFrame({
        'window_length': np.concatenate(window_lengths) * frame_time / 60,
        'increase_rate': np.concatenate(increase_rates) * 60 / frame_time,
        'step_size_during': np.concatenate(step_sizes_during) * scale,
        'peak_volume': np.concatenate(peak_volumes) * scale**3,
        'spot_density': np.concatenate(spot_densities) / scale**3,
        'circularity': np.concatenate(circularities),
        'peak_intensity': np.concatenate(peak_intensities),
        'time_to_budding': np.concatenate(times_to_budding) * frame_time / 60,
        'step_size_after': np.concatenate(step_sizes_after) * scale,
        'cytosol_intensity': np.concatenate(cytosol_intensities),
        'cell_size': np.concatenate(cell_sizes) * scale**2,
        'replicate': np.concatenate(replicates),
        'dataset': np.concatenate(labels)
    })

    return df