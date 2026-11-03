import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.signal import savgol_filter

from lib.prepare_properties import create_property_dataframe


scale = 63.69/512 # um/pixel
frame_time = 20 # seconds per frame


directories = [directory_spa2_wt, directory_spa2_bem1d_succ, directory_spa2_3del, directory_spa2_A1, directory_spa2_A10, directory_spa2_T3,
               directory_gic2_wt, directory_gic2_3del]
dataset_labels = ['Spa2 (ancestor)', r'Spa2 (bem1$\Delta$)', 'Spa2 (REM)', 'Spa2 (EV1)', 'Spa2 (EV2)', 'Spa2 (EV3)', 'Gic2PBD (ancestor)', 'Gic2PBD (REM)']
color_palette = ['#990000', '#7a0177', '#d7301f', '#ef6548', '#fc8d59', '#fdbb84', '#005824', '#66c2a4']
markers = ['o', 'o', 'o', 'o', 'o', 'o', '^', '^']
linestyles = ['-', '-', '-', '-', '-', '-', '--', '--']

property_labels = [r'$\tau_e$', r'$r$', r'$\Delta s_e$',
                    r'$V$', r'$\rho$', r'$c$',
                    r'$\tau_m$']


indices = np.arange(1, len(directories)+1, 1)
for directory, label, i in zip(directories, dataset_labels, indices):
    df = create_property_dataframe([directory], label, frame_time=frame_time, scale=scale)
    df = df[['window_length', 'increase_rate', 'step_size_during', 'peak_volume', 'spot_density', 'circularity', 'time_to_budding']]     # select columns
    data = np.array(df).T

    ## UNPACK DATA IN ARRAY AND ATTACH LABELS AND COLORS FOR PLOTTING
    if i == 1:
        all_data = data
        colors = [color_palette[i-1] for _ in range(len(data.T))]
        labels = [label for _ in range(len(data.T))]
        continue
    if 1 < i < len(directories):
        all_data = np.hstack((all_data, data))
        for _ in range(len(data.T)):
            colors.append(color_palette[i-1])
            labels.append(label)
        continue
    if i == len(directories):
        all_data = np.hstack((all_data, data))
        for _ in range(len(data.T)):
            colors.append(color_palette[i-1])
            labels.append(label)

colors = np.array(colors)
labels = np.array(labels)

## REMOVE CELLS THAT HAVE NAN'S FOR ONE OR MORE OBSERVABLES
all_data_clean = all_data[:, ~np.isnan(all_data).any(axis=0)]
print(f'Cells lost to nan\'s: {all_data.shape[1] - all_data_clean.shape[1]}')
colors = colors[~np.isnan(all_data).any(axis=0)]
labels = labels[~np.isnan(all_data).any(axis=0)]


## STANDARDIZE DATA FOR PCA (MEAN = 0, STD = 1)
std_scaler = StandardScaler()
all_data_standardized = std_scaler.fit_transform(all_data_clean.T)  # STANDARD_SCALER ASSUMES (n_samples, n_features)

## PERFORM PCA AND TRANSFORM
pca = PCA()
x_new = pca.fit_transform(all_data_standardized)
ratios = pca.explained_variance_ratio_
eigenvectors_sorted = pca.components_

print('Explained variance per eigenvector:')
print(ratios)

print('Eigenvectors:')
print(eigenvectors_sorted)
np.savetxt('C:/Data/pca_eigenvectors.txt', eigenvectors_sorted, fmt='%.2f')

## PLOT EXPLAINED VARIANCE/SCREE PLOT
sum_ratios = np.cumsum(ratios)
fig, ax = plt.subplots(figsize=(4, 2.2))
ax.plot(sum_ratios, c='k', marker='o', markersize=3)
ax.hlines(0.8, ax.get_xlim()[0], ax.get_xlim()[1], color='darkred', ls='--')
ax.set_ylabel('Fraction of explained \nvariance')
ax.set_xticks(range(len(property_labels)))
ax.set_xticklabels([f'PC{i}' for i in range(1, len(property_labels)+1)])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim([0,1.1])
plt.tight_layout()

## DETERMINE NUMBER OF COMPONENTS NEEDED FOR > 80% EXPLAINED VARIANCE
cutoff = 0
for i in range(len(sum_ratios)):
    if sum_ratios[i] > 0.8:
        cutoff = i
        break

contributions = np.zeros((cutoff+1, len(property_labels)))
for i in range(cutoff+1):
    eigen = eigenvectors_sorted[i]
    for j in range(len(property_labels)):
        contributions[i,j] = eigen[j]**2


## PLOT PC'S AND COMPOSITIONS
sns.set_style('white')
fig, ax = plt.subplots(figsize=(2.2, 2.5))
bottom = np.zeros(cutoff+1)
x = np.arange(cutoff+1)
color_palette_bars = sns.color_palette('deep')
for i in range(len(property_labels)-1, -1, -1):
    bars = ax.bar(x, contributions[:,i], bottom=bottom, label=property_labels[i], color=color_palette_bars[i])
    bottom += contributions[:,i]
    ev_row = eigenvectors_sorted[0:cutoff+1, i]
    bar_labels = []
    for j in range(len(ev_row)):
        if ev_row[j] > 0:
            bar_labels.append('+')
        else:
            bar_labels.append('-')
    ax.bar_label(bars, labels=bar_labels, label_type='center', fontsize=20)
ax.set_xticks(range(cutoff+1))
ax.set_xticklabels([f'PC{i}' for i in range(1, cutoff+2)])
ax.set_ylabel('Relative contribution')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
handles, legend_labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], legend_labels[::-1], frameon=False)
plt.tight_layout()


## PLOT DENSITY ON PC1
conds = dataset_labels
sns.set_style('ticks')
fig = plt.figure(figsize=(3, 2))
ax = fig.add_subplot(1, 1, 1)
for i, c in enumerate(conds):
    d = x_new[labels ==c]
    heights, bins = np.histogram(d[:,0], bins=np.linspace(-4, 4, 13), density=True)
    bin_width = np.diff(bins)[0]
    bin_pos = (bins[:-1] + bin_width / 2)
    ax.plot(bin_pos, heights, color=color_palette[i], lw=1.5, label=dataset_labels[i], ls=linestyles[i])
ax.set_xlabel('PC1')
ax.set_ylabel('Density')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()


## PLOT OTHER PC PROJECTIONS
fig = plt.figure(figsize=(7, 2))
ax = fig.add_subplot(1, 4, 1)
for i, c in enumerate(conds):
    d = x_new[labels ==c]
    mean = d.mean(axis=0)
    err = d.std(axis=0) / np.sqrt(len(d))
    ax.errorbar(mean[1], mean[2], xerr=err[1], yerr=err[2], capsize=4, label=dataset_labels[i], color=color_palette[i], marker=markers[i], ms=6.5, ecolor='k')
    ax.set_xlabel('PC2')
    ax.set_ylabel('PC3')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xticks(np.arange(-1, 2, 1))
    ax.set_yticks(np.arange(-1, 2, 1))

ax = fig.add_subplot(1, 4, 2)
for i, c in enumerate(conds):
    d = x_new[labels ==c]
    mean = d.mean(axis=0)
    err = d.std(axis=0) / np.sqrt(len(d))
    ax.errorbar(mean[2], mean[3], xerr=err[2], yerr=err[3], capsize=4, label=dataset_labels[i], color=color_palette[i], marker=markers[i], ms=6.5, ecolor='k')
    ax.set_xlabel('PC3')
    ax.set_ylabel('PC4')
    ax.legend(frameon=False)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xticks(np.arange(-1, 2, 1))
    ax.set_yticks(np.arange(-1, 2, 1))

ax = fig.add_subplot(1, 4, 3)
for i, c in enumerate(conds):
    d = x_new[labels ==c]
    mean = d.mean(axis=0)
    err = d.std(axis=0) / np.sqrt(len(d))
    ax.errorbar(mean[1], mean[3], xerr=err[1], yerr=err[3], capsize=4, label=dataset_labels[i], color=color_palette[i], marker=markers[i], ms=6.5, ecolor='k')
    ax.set_xlabel('PC2')
    ax.set_ylabel('PC4')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xticks(np.arange(-1, 2, 1))
    ax.set_yticks(np.arange(-1, 2, 1))

plt.tight_layout()
plt.show()


