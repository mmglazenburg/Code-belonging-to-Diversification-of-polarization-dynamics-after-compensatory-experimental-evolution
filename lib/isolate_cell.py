import numpy as np
import math
from skimage import measure


def crop_cell(image, mask, cell_id, return_mask=False):
    """
    Crop a single cell from a 2D image frame using a segmentation mask.

    The selected cell is isolated by setting all other pixels to zero,
    and the image is cropped to a bounding box around the cell.

    Parameters
    ----------
    image : ndarray
        2D image array.
    mask : ndarray
        Segmentation mask where each cell has a unique integer label.
    cell_id : int
        Label corresponding to the cell of interest.
    return_mask : bool, optional
        If True, also return the cropped mask.

    Returns
    -------
    ndarray
        Cropped image containing only the selected cell.
    ndarray (optional)
        Cropped mask (if return_mask=True).
    """

    # Convert mask to binary (1 = cell of interest, 0 = background)
    mask = (mask == cell_id).astype(np.uint8)

    # Apply mask to image
    image = np.multiply(mask, image)

    # Find cell contour
    contour = measure.find_contours(mask, 0.5)
    if len(contour) < 1:
        error_string = "Cell ID " + str(cell_id) + " has no mask in one or more frames"
        raise Exception(error_string)

    contour = contour[0]

    # Compute bounding box (with small padding)
    min_x = math.ceil(np.min(contour[:, 1]) - 1)
    max_x = math.floor(np.max(contour[:, 1]) + 2)
    min_y = math.ceil(np.min(contour[:, 0]) - 1)
    max_y = math.floor(np.max(contour[:, 0]) + 2)

    # Crop image and mask
    single_cell_tiff = image[min_y:max_y, min_x:max_x]
    mask_cropped = mask[min_y:max_y, min_x:max_x]

    if return_mask:
        return single_cell_tiff, mask_cropped
    else:
        return single_cell_tiff


def crop_zstack(image, mask, cell_id):
    """
    Crop a 3D z-stack using a 2D segmentation mask.

    The cropping window is determined from a single slice and then
    applied consistently across all z-slices.

    Parameters
    ----------
    image : ndarray
        3D image array with shape (z, height, width).
    mask : ndarray
        2D segmentation mask (height, width).
    cell_id : int
        Label corresponding to the cell of interest.

    Returns
    -------
    ndarray
        Cropped 3D image stack.
    """

    # Determine crop size from first slice
    mid_slice = crop_cell(image[0], mask, cell_id)
    h, w = mid_slice.shape

    # Initialize cropped stack
    cropped_stack = np.zeros((len(image), h, w))

    # Apply identical cropping to all slices
    for i in range(len(image)):
        cropped_stack[i] = crop_cell(image[i], mask, cell_id)

    return cropped_stack