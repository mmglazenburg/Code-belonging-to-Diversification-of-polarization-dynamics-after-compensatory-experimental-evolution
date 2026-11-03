import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# Ensure text remains editable in SVG export
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

from lib.prepare_properties import create_property_dataframe

# =========================
# SELECT DATASETS
# =========================
# Choose ONE comparison by uncommenting

# Example 1: Effect of denoising
# directories = ["path_to_dataset_1/", "path_to_dataset_2/"]
# dataset_labels = ['With Denoise.ai', 'Without Denoise.ai']
# palette = sns.color_palette('Paired')

# Example 2: Mutant comparison
# directories = ["path_to_dataset_1/", "path_to_dataset_2/", "path_to_dataset_3/"]
# dataset_labels = ['WT', r'rsr1Δ', r'axl2Δ rax1Δ']
# palette = sns.color_palette('Set2')

# Example 3: Different proteins
# directories = ["path_to_dataset_1/", "path_to_dataset_2/", "path_to_dataset_3/"]
# dataset_labels = ['Cdc42', 'Gic2PBD', 'Spa2']
# palette = [sns.color_palette('Set2')[0],
#            sns.color_palette('Set2')[5],
#            sns.color_palette('Set2')[6]]

# Example 4: Single comparison
# directories = ["path_to_dataset_1/", "path_to_dataset_2/"]
# dataset_labels = ['Condition A', 'Condition B']
# palette = sns.color_palette('Set1')

# ---- ACTIVE CONFIGURATION ----
directories = [
    "path_to_dataset_1/",
    "path_to_dataset_2/"
]

dataset_labels = ['Condition A', 'Condition B']
palette = sns.color_palette('Paired')

# =========================
# PARAMETERS
# =========================
scale = 63.69 / 512   # µm per pixel
frame_time = 20       # seconds per frame

# =========================
# LOAD DATA
# =========================
df = create_property_dataframe(
    directories,
    dataset_labels,
    frame_time=frame_time,
    scale=scale
)

# Reshape for plotting (compare establishment vs maintenance)
df_melted = df.melt(
    id_vars='dataset',
    value_vars=['step_size_during', 'step_size_after'],
    var_name='phase',
    value_name='step_sizes'
)

# =========================
# PLOTTING
# =========================
sns.set_theme('paper')
sns.set_style('whitegrid')

plt.figure(figsize=(4, 3))

ax = sns.violinplot(
    data=df_melted,
    y='step_sizes',
    x='phase',
    hue='dataset',
    legend=True,
    palette=palette,
    inner='quart',
    cut=1,
    linewidth=1,
    alpha=0.9,
    density_norm='width'
)

sns.stripplot(
    data=df_melted,
    y='step_sizes',
    x='phase',
    hue='dataset',
    palette='dark:k',
    marker='.',
    size=3.5,
    alpha=0.8,
    dodge=True,
    legend=False
)

# =========================
# FORMATTING
# =========================
plt.ylabel(r'Spot mobility $\Delta s$ (µm/20 s)')
plt.ticklabel_format(style='sci', axis='y', scilimits=(-2, 2))
plt.xlabel('')

# Rename x-axis categories
locs, _ = plt.xticks()
plt.xticks(locs, labels=['Establishment', 'Maintenance'])

# Legend formatting
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, title='')

# =========================
# STATISTICAL ANNOTATIONS
# =========================

pairs = []

# Compare establishment vs maintenance within each dataset
for dataset_label in dataset_labels:
    pairs.append([
        ('step_size_during', dataset_label),
        ('step_size_after', dataset_label)
    ])

if len(dataset_labels) > 1:
    # Compare datasets within each phase
    for i in range(1, len(dataset_labels)):
        pairs.append([
            ('step_size_during', dataset_labels[0]),
            ('step_size_during', dataset_labels[i])
        ])

    for i in range(1, len(dataset_labels)):
        pairs.append([
            ('step_size_after', dataset_labels[0]),
            ('step_size_after', dataset_labels[i])
        ])

annotator = Annotator(
    ax,
    pairs,
    data=df_melted,
    x='phase',
    y='step_sizes',
    hue='dataset'
)

annotator.configure(test='Mann-Whitney', text_format='star')
annotator.apply_and_annotate()

# =========================
# DISPLAY
# =========================
plt.tight_layout()
plt.show()