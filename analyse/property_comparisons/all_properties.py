import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import scienceplots

# Ensure text remains editable in SVG export
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

from lib.utilities import get_signal_start_end
from lib.prepare_properties import create_property_dataframe

# =========================
# SELECT DATASET
# =========================
# Choose ONE dataset by setting the directory

# Example 1
directory = "path_to_dataset/"

# Example 2
# directory = "path_to_other_dataset/"

# =========================
# PARAMETERS
# =========================
scale = 63.69 / 512   # µm per pixel
frame_time = 20       # seconds per frame

# Derived paths
directory_cells = directory + 'cells/'

# =========================
# LOAD DATA
# =========================
all_intensity_curves = np.loadtxt(directory + 'all_intensity_traces.txt')

df = create_property_dataframe(
    [directory],
    ['Dataset'],
    frame_time=frame_time,
    scale=scale
)

# =========================
# PLOTTING SETUP
# =========================
sns.set_theme('paper', font_scale=1)
sns.set_style('ticks')

# Colors for different stages
colors = ['#fdae61', '#ffffbf', '#abd9e9']
intensity_color = 'k'

plt.figure(figsize=(4.7, 7))

# =========================
# INTENSITY TRACE PANEL
# =========================
plt.subplot(4, 1, 1)

means = np.nanmean(all_intensity_curves, axis=0)
means_start, means_end = get_signal_start_end(means)

time = np.arange(len(means)) * frame_time / 60
time = time[::-1]   # Time until bud emergence (0 at end)

# Plot individual traces
for intensity in all_intensity_curves:
    plt.plot(time, intensity, c=intensity_color, alpha=0.1, lw=0.5)

# Plot mean trace
plt.plot(time, means, lw=2, c=intensity_color, label='Mean intensity')

# Dummy for legend
plt.plot(time, np.full(len(time), np.nan),
         c='k', alpha=0.15, lw=0.5,
         label='Individual traces')

plt.xlabel('Time until bud emergence (min)')
plt.gca().xaxis.set_inverted(True)
plt.ylabel('Normalized intensity')
plt.ticklabel_format(axis='y', style='sci', scilimits=[0, 0])

# =========================
# POLARIZATION STAGES
# =========================
alpha_stages = 0.5
max_y = plt.gca().get_ylim()[1]

# Early stage
plt.axvspan(time[0] - means_start / 3,
            time[0] - (means_end - 2) / 3,
            0, 1, alpha=alpha_stages, color=colors[0])

# Transition
plt.axvspan(time[0] - (means_end - 2) / 3,
            time[0] - (means_end + 2) / 3,
            0, 1, alpha=alpha_stages, color=colors[1])

# Late stage
plt.axvspan(time[0] - (means_end + 2) / 3,
            time[-1],
            0, 1, alpha=alpha_stages, color=colors[2])

sns.set_style('whitegrid')

# =========================
# VIOLIN PLOTS
# =========================
alpha_violins = 0.9
swarm_marker_size = 1.75
swarm_alpha = 0.9
label_pad = 1

# ---- ESTABLISHMENT PHASE ----
plt.subplot(4, 3, 4)
ax = sns.violinplot(data=df, y='window_length',
                    color=colors[0], alpha=alpha_violins, inner='quart')
sns.stripplot(data=df, y='window_length',
              color='k', size=swarm_marker_size,
              alpha=swarm_alpha)
ax.set_ylabel(r"$\tau_e$ (min)")
plt.ticklabel_format(axis='y', style='sci', scilimits=[-2, 2])
ax.set_ylim([0, ax.get_ylim()[1]])

plt.subplot(4, 3, 5)
ax = sns.violinplot(data=df, y='increase_rate',
                    color=colors[0], alpha=alpha_violins, inner='quart')
sns.stripplot(data=df, y='increase_rate',
              color='k', size=swarm_marker_size,
              alpha=swarm_alpha)
ax.set_ylabel(r"$r$ (/min)")
plt.ticklabel_format(axis='y', style='sci', scilimits=[-2, 2])

plt.subplot(4, 3, 6)
ax = sns.violinplot(data=df, y='step_size_during',
                    color=colors[0], alpha=alpha_violins, inner='quart')
sns.stripplot(data=df, y='step_size_during',
              color='k', size=swarm_marker_size,
              alpha=swarm_alpha)
ax.set_ylabel(r"$\Delta s_e$ (µm/20 s)")
plt.ticklabel_format(axis='y', style='sci', scilimits=[-2, 2])
ylim_step_size = ax.get_ylim()[1]
ax.set_ylim([0, ylim_step_size])

# ---- PEAK PROPERTIES ----
plt.subplot(4, 3, 7)
ax = sns.violinplot(data=df, y='peak_volume',
                    color=colors[1], alpha=alpha_violins, inner='quart')
sns.stripplot(data=df, y='peak_volume',
              color='k', size=swarm_marker_size,
              alpha=swarm_alpha)
ax.set_ylabel(r"$V$ (µm$^3$)")

plt.subplot(4, 3, 8)
ax = sns.violinplot(data=df, y='circularity',
                    color=colors[1], alpha=alpha_violins, inner='quart')
sns.stripplot(data=df, y='circularity',
              color='k', size=swarm_marker_size,
              alpha=swarm_alpha)
ax.set_ylabel(r"$c$")
ax.set_ylim([0.5, 1])

plt.subplot(4, 3, 9)
ax = sns.violinplot(data=df, y='spot_density',
                    color=colors[1], alpha=alpha_violins, inner='quart')
sns.stripplot(data=df, y='spot_density',
              color='k', size=swarm_marker_size,
              alpha=swarm_alpha)
ax.set_ylabel(r"$\rho$ (µm$^{-3}$)")

# ---- MAINTENANCE PHASE ----
plt.subplot(4, 3, 10)
ax = sns.violinplot(data=df, y='step_size_after',
                    color=colors[2], alpha=alpha_violins, inner='quart')
sns.stripplot(data=df, y='step_size_after',
              color='k', size=swarm_marker_size,
              alpha=swarm_alpha)
ax.set_ylabel(r"$\Delta s_m$ (µm/20 s)")
ax.set_ylim([0, ylim_step_size])

plt.subplot(4, 3, 11)
ax = sns.violinplot(data=df, y='time_to_budding',
                    color=colors[2], alpha=alpha_violins, inner='quart')
sns.stripplot(data=df, y='time_to_budding',
              color='k', size=swarm_marker_size,
              alpha=swarm_alpha)
ax.set_ylabel(r"$\tau_m$ (min)")

# =========================
# DISPLAY
# =========================
plt.tight_layout(w_pad=0.5)
plt.show()