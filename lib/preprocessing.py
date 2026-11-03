import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def exp(x, c1, c2, tau):
    """
    Exponential decay function used for bleaching correction.

    Parameters
    ----------
    x : array-like
        Time points.
    c1 : float
        Offset (baseline intensity).
    c2 : float
        Amplitude.
    tau : float
        Decay constant.

    Returns
    -------
    array-like
        Evaluated exponential function.
    """
    return c1 + c2 * np.exp(-x / tau)


def correct_bleaching(image, mask, plot_fit=False, print_coeffs=False):
    """
    Correct photobleaching in a time-lapse 3D image.

    The correction is based on fitting an exponential decay to the average
    intensity over time (computed from masked max projections), and then
    normalizing each frame accordingly.

    Parameters
    ----------
    image : ndarray
        4D image array (t, z, y, x).
    mask : ndarray
        3D mask (t, y, x) with labeled cell regions.
    plot_fit : bool, optional
        If True, display the bleaching curve and fitted model.
    print_coeffs : bool, optional
        If True, print fitted exponential parameters.

    Returns
    -------
    ndarray
        Bleaching-corrected image.
    """

    # =========================
    # CREATE MAX PROJECTION
    # =========================
    mask_cells = np.zeros_like(mask)
    mask_cells[mask > 0] = 1

    image_project = np.max(image, axis=1) * mask_cells
    image_project = image_project.astype(float)

    # Remove background (set zeros to NaN)
    image_project[image_project == 0] = np.nan

    # =========================
    # COMPUTE AVERAGE INTENSITY
    # =========================
    avg_intensity = np.nanmean(image_project, axis=(1, 2))

    # =========================
    # FIT EXPONENTIAL DECAY
    # =========================
    x = np.arange(0, len(avg_intensity), 1)

    # Initial guess: [offset, amplitude, decay time]
    p0 = [avg_intensity[-1], avg_intensity[0] - avg_intensity[-1], 1]

    popt, pcov = curve_fit(exp, x, avg_intensity, p0=p0)

    if print_coeffs:
        print(popt)

    bleaching_curve = exp(x, *popt)

    # =========================
    # OPTIONAL PLOTTING
    # =========================
    if plot_fit:
        plt.figure()
        plt.plot(avg_intensity, label='Data')
        plt.plot(bleaching_curve, label='Fit')
        plt.title('Bleaching curve')
        plt.xlabel('Time point [frames]')
        plt.ylabel('Average intensity per frame [a.u.]')
        plt.legend()
        plt.show()

    # =========================
    # NORMALIZE CURVE
    # =========================
    bleaching_curve = bleaching_curve / bleaching_curve[0]

    # =========================
    # APPLY CORRECTION
    # =========================
    image_corrected = np.zeros_like(image)

    for i in range(len(bleaching_curve)):
        image_corrected[i] = image[i] / bleaching_curve[i]

    return image_corrected


def check_mask(mask, cell_info):
    """
    Verify that all specified cell IDs exist in the mask across required frames.

    This prevents downstream errors caused by missing segmentation labels.

    Parameters
    ----------
    mask : ndarray
        3D array (t, y, x) containing segmentation mask.
    cell_info : ndarray
        Array with columns [cell_id, start_frame, end_frame].
    """

    error_found = False

    for line in range(len(cell_info)):
        cell_id = cell_info[line, 0]
        timerange = range(cell_info[line, 1], cell_info[line, 2])

        for t in timerange:
            if cell_id not in mask[t]:
                print('Mask error for cell', cell_id, 'in frame', t + 1)
                error_found = True

    if error_found:
        raise ValueError(
            'Mask errors found: one or more cell IDs do not exist in one or more frames'
        )