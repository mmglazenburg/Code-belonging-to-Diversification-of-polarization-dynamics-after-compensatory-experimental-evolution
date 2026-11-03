import numpy as np
import pandas as pd
import scipy as sc


def list_to_array(array_list):
    """
    Convert a list of arrays with unequal shapes into a zero-padded array.

    Parameters
    ----------
    array_list : list of ndarray
        List of arrays with the same number of dimensions but varying sizes.

    Returns
    -------
    ndarray
        Array where all input arrays are padded with zeros to match
        the largest shape along each dimension.
    """

    # Determine maximum size along each dimension
    dimensions = np.zeros(array_list[0].ndim, dtype=int)

    for array in array_list:
        shape = array.shape
        for i in range(len(shape)):
            if shape[i] > dimensions[i]:
                dimensions[i] = shape[i]

    # Initialize padded array
    padded_arrays = np.zeros((len(array_list), *tuple(dimensions)), dtype=np.float32)

    # Pad each array to match maximum dimensions
    for i, array in enumerate(array_list):
        pad_widths = [(0, dimensions[j] - array.shape[j]) for j in range(len(dimensions))]
        padded_arrays[i] = np.pad(array, tuple(pad_widths))

    return padded_arrays


def align_traces(trace_list, pstarts=None, fraction_req=0.5):
    """
    Align time traces based on the detected start of polarity.

    Parameters
    ----------
    trace_list : list of ndarray
        List of time series.
    pstarts : array-like, optional
        Precomputed start indices of polarity for each trace.
        If None, they are computed automatically.
    fraction_req : float
        Minimum fraction of traces required at a timepoint
        for it to be retained after alignment.

    Returns
    -------
    ndarray
        Aligned and trimmed traces.
    """

    if pstarts is None:
        pstarts = np.zeros(len(trace_list))
        for i, trace in enumerate(trace_list):
            pstart, _ = get_signal_start_end(trace)
            pstarts[i] = pstart

    # Determine alignment parameters
    pstart_max = np.max(pstarts)
    tail_lengths = [len(trace) - pstarts[i] for i, trace in enumerate(trace_list)]
    tail_length_max = max(tail_lengths)
    tail_length_avg = sum(tail_lengths) / len(tail_lengths)

    traces_padded = []

    for i, trace in enumerate(trace_list):
        diff_start = int(pstart_max - pstarts[i])
        diff_end = int(tail_length_max - tail_lengths[i])

        trace_padded = np.pad(
            trace,
            (diff_start, diff_end),
            constant_values=(np.nan, np.nan)
        )
        traces_padded.append(trace_padded)

    traces_matrix = np.array(traces_padded)

    # Determine fraction of valid traces per timepoint
    traces_present = np.where(np.isnan(traces_matrix), 0, 1)
    fraction_present = np.sum(traces_present, axis=0) / len(traces_present)

    # Find starting index with sufficient data coverage
    for i in range(len(fraction_present)):
        if fraction_present[i] > fraction_req:
            i_start = i
            break

    # Define stop index based on average tail length
    i_stop = int(pstart_max + tail_length_avg)

    return traces_matrix[:, i_start:i_stop]


def get_signal_start_end(signal, grad_threshold=0.1, signal_threshold=0.7):
    """
    Determine start and end of a polarization signal.

    Parameters
    ----------
    signal : ndarray
        1D signal trace.
    grad_threshold : float
        Fraction of maximum gradient used to detect plateau.
    signal_threshold : float
        Fraction of maximum signal used to confirm peak.

    Returns
    -------
    tuple
        (start_index, end_index)
    """

    # --- Determine start ---
    polar_start = 0
    filtered_grad = np.gradient(signal)

    for t in range(np.argmax(signal), -1, -1):
        if (
            filtered_grad[t] < 0.1 * np.max(filtered_grad)
            and signal[t] < 0.15 * np.max(signal)
        ):
            polar_start = t
            break
        elif t == 0:
            print("Warning: polar_start could not be determined correctly")

    # --- Determine end ---
    polar_end = len(signal) - 1

    try:
        tail_start = polar_start
        filtered_grad_tail = filtered_grad[tail_start:]

        if np.all(filtered_grad_tail > grad_threshold * np.max(filtered_grad)):
            polar_end = np.argmin(filtered_grad_tail) + tail_start
            print("Warning: polar_end could not be determined correctly")
        else:
            for t in range(len(filtered_grad_tail)):
                if (
                    filtered_grad_tail[t] < grad_threshold * np.max(filtered_grad)
                    and signal[t + tail_start] > signal_threshold * np.max(signal)
                ):
                    polar_end = t + tail_start
                    break

    except ValueError:
        polar_end = len(signal) - 1

    return polar_start, polar_end

