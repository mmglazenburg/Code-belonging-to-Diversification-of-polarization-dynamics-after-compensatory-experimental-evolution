import scipy as sc
import numpy as np
import skimage.filters


def interpolate_z(cell, scale, z_width):
    """
    Rescale the z-dimension of a 3D image using interpolation so that voxel
    spacing becomes isotropic (same in x, y, and z).

    Parameters
    ----------
    cell : ndarray
        3D image array with shape (z, x, y).
    scale : float
        Pixel size in x and y (µm per pixel).
    z_width : float
        Distance between z-slices (µm).

    Returns
    -------
    ndarray
        Interpolated 3D image with rescaled z-axis.
    """

    num_slices = len(cell)
    pixels_per_slice = z_width / scale

    rescaled_z_num = int(pixels_per_slice * (num_slices - 1))

    z_dim, x_dim, y_dim = cell.shape

    x = np.linspace(0, x_dim - 1, x_dim)
    y = np.linspace(0, y_dim - 1, y_dim)
    z = np.linspace(0, z_dim - 1, z_dim)
    z_fine = np.linspace(0, z_dim - 1, rescaled_z_num)

    interp = sc.interpolate.RegularGridInterpolator((z, x, y), cell)

    Z, X, Y = np.meshgrid(z_fine, x, y, indexing='ij')

    return interp((Z, X, Y))


def normalize_image(image):
    """
    Normalize a timelapse image using the average intensity of the
    unpolarized state (first 5 frames).

    A max projection is used to avoid averaging over empty/background regions.

    Parameters
    ----------
    image : ndarray
        Image array (t, z, x, y) or (t, x, y).

    Returns
    -------
    ndarray
        Normalized image.
    """

    image_start = image[0:5]

    if image.ndim > 3:
        image_start_max = np.max(image_start, axis=1)
    else:
        image_start_max = image_start

    image_start_max = image_start_max[image_start_max > 0]

    normalization = np.mean(image_start_max)

    return image / normalization


def variance_threshold(image, std_no=3):
    """
    Compute an intensity threshold based on the first 5 frames of the movie.

    The threshold is defined as:
        mean + std_no * standard deviation

    Parameters
    ----------
    image : ndarray
        Input image or timelapse.
    std_no : float
        Number of standard deviations above the mean.

    Returns
    -------
    float
        Threshold value.
    """

    image_start = image[0:5]

    if image.ndim > 3:
        image_start_max = np.max(image_start, axis=1)
    else:
        image_start_max = image_start

    image_start_max = image_start_max[image_start_max > 0]

    threshold = np.mean(image_start_max) + std_no * np.std(image_start_max)

    return threshold


def mask_polarity(image, threshold, sigma_xyz=1):
    """
    Segment polarity signal by applying a threshold to a Gaussian-smoothed image.

    Parameters
    ----------
    image : ndarray
        Timelapse image (3D or 4D).
    threshold : float
        Intensity threshold for segmentation.
    sigma_xyz : float
        Standard deviation for Gaussian smoothing in spatial dimensions.

    Returns
    -------
    ndarray
        Masked image where values below threshold are set to zero.
    """

    if image.ndim == 3:
        sigma = (0, sigma_xyz, sigma_xyz)
    else:
        sigma = (0, sigma_xyz, sigma_xyz, sigma_xyz)

    image_filtered = skimage.filters.gaussian(image, sigma=sigma)
    mask = image_filtered > threshold

    return image * mask