import numpy as np
import scipy as sc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns


def relabel_spot_mask(mask):
    """
    Relabel a spot mask so that labels are contiguous:
    0 stays background, spots become 1..N
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]  # drop background

    new_mask = np.zeros_like(mask, dtype=int)

    for new_label, old_label in enumerate(unique_labels, start=1):
        new_mask[mask == old_label] = new_label

    return new_mask


def make_clustered_mask(masked_image, plot=False, clustering_distance_threshold=15):
    """
    Take an image where the polar spot has been masked out (i.e. all signal outside polar zone is zero) and use
    a clustering algorithm to distinguish separate spots. Separate objects are identified and center of masses
    of all objects in the timelapse are clustered using agglomerative clustering. The cluster mapping of the center
    of masses is then mapped back onto the objects. What is returned is a discrete mask where each spot has a unique
    integer identifier and the background is zero.
    :param masked_image: 3D timelapse of thresholded cell, where all non-polarity signal is zero
    :param plot: create a plot of the identified clusters and their intensity traces if True
    :param clustering_distance_threshold: parameter to adjust agglomerative clustering strictness
    :return: 3D timelapse of mask identifying each separate spot
    """

    label_frames = []
    coms = []
    for frame in masked_image:
        ## IDENTIFY SPATIALLY SEPARATE OBJECTS AND CALCULATE COM
        labels, num_features = sc.ndimage.label(frame)
        com = sc.ndimage.center_of_mass(frame, labels, index=range(1, num_features + 1))
        coms.append(com)
        label_frames.append(labels)

    com_list = []

    ## FLATTEN COM LIST
    for set in coms:
        for com in set:
            com_list.append(com)
    com_array = np.array(com_list)

    ## APPLY CLUSTERING ALGORITHM AND EXTRACT LABELS, I.E. CLUSTER LABEL FOR EACH COM
    agg = AgglomerativeClustering(distance_threshold=clustering_distance_threshold, n_clusters=None, linkage='average').fit(com_array)
    labels = agg.labels_

    if plot:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1, projection='3d')

        # ax.set_title('Agglomerative clustering')
        colors = sns.color_palette('Dark2')
        n_clusters_ = np.max(labels) + 1
        for k, col in zip(range(n_clusters_), colors):
            class_members = labels == k
            ax.scatter(com_array[class_members, 0], com_array[class_members, 1], com_array[class_members, 2], color=col, marker=".")
        ax.set_xlim(0, masked_image.shape[1])
        ax.set_ylim(0, masked_image.shape[2])
        ax.set_zlim(0, masked_image.shape[3])

    clustered_labels = []
    i = 0
    for label_frame in label_frames:
        ## FOR EACH LABELED IMAGE FRAME THAT HAS OBJECTS, ASSIGN THE IDENTIFIED CLUSTER LABELS TO THOSE OBJECTS
        if np.max(label_frame) != 0:
            for j in range(1, np.max(label_frame) + 1):
                label_frame[label_frame == j] = labels[i] + 1
                i += 1
        clustered_labels.append(label_frame)

    clustered_labels = np.array(clustered_labels, dtype="uint8")

    if plot:
        ax = fig.add_subplot(1, 2, 2)

    ## FIND THE SPOT WITH THE HIGHEST INTENSITY
    max_spot_intensity = 0
    for i in range(1, np.max(clustered_labels)+1):
        current_mask = clustered_labels == i
        image_masked = masked_image * current_mask

        intensity = np.sum(image_masked, axis=(1, 2, 3))
        if np.max(intensity) > max_spot_intensity:
            max_spot_intensity = np.max(intensity)

    for i in range(1, np.max(clustered_labels) + 1):
        current_mask = clustered_labels == i
        image_masked = masked_image * current_mask

        intensity = np.sum(image_masked, axis=(1, 2, 3))
        excluded = ''
        ## SET SPOTS THAT ARE LESS THAN 10% OF THE MAX INTENSITY TO ZERO IN THE MASK; CHECK AFTER FRAME 5 TO AVOID CYTOKINESIS
        if np.max(intensity[5:]) < 0.1 *max_spot_intensity:
            clustered_labels[current_mask] = 0
            excluded = ' (excluded)'

        total_sum = np.sum(intensity[5:])

        if plot:
            filter_window = max(int(len(intensity) / 5), 4)  # TAKE POLY_ORDER + 1 IF WINDOW IS SMALLER
            filtered = sc.signal.savgol_filter(intensity, filter_window, 3)
            time = np.arange(len(intensity)) / 3
            ax.scatter(time, intensity, color=colors[i-1], s=6)
            label = "Tot = %.2E" %total_sum + excluded
            ax.plot(time, filtered, c=colors[i-1], label=label)
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Intensity (a.u.)')
    if plot:
        plt.legend()
        plt.tight_layout()
        plt.show()

    clustered_labels = relabel_spot_mask(clustered_labels).astype(np.uint16)
    return clustered_labels


def determine_final_cluster_index(image, cluster_mask):
    """
    Takes an image and its clustering mask and returns the index of the cluster that is dominant in the final 4 frames
    of the timelapse, i.e. the winning spot.
    :param image: 3d timelapse of a polarizing cell
    :param cluster_mask: mask with the same shape as image where each separate spot is identified by a different integer
    :return: index corresponding to the winning cluster
    """

    intensities = []
    for i in range(1, np.max(cluster_mask) + 1):
        current_mask = cluster_mask == i
        image_masked = image * current_mask

        intensity = np.sum(image_masked, axis=(1, 2, 3))
        intensities.append(intensity)

    intensities = np.array(intensities)

    intensities_final = intensities[:, -4:]
    mean_final_intensities = np.mean(intensities_final, axis=1)
    final_index = np.argmax(mean_final_intensities)

    return final_index + 1
