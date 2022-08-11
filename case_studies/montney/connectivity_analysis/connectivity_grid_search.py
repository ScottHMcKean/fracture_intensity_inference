# File to run connectivity analysis in a similar fashion to PEST
# Uses a .ptf file, but instead of PEST perturbing it, we do it in Python

import numpy as np
import os
import subprocess
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# make sure we are in the right directory
os.chdir(
    r"C:\repos\fracture_intensity_inference\case_studies\montney\connectivity_analysis"
)

# setup the grid
# setup the grid
p32_search = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 10e-3]
min_length_search = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500]

p32s, min_lengths = np.meshgrid(p32_search, min_length_search, sparse=False)
search_runs = np.array([p32s, min_lengths]).reshape(2, -1).T

# run everything
for i in range(0, search_runs.shape[0]):
    print(i, search_runs[i, 0], search_runs[i, 1])
    cmd = f"python run_connectivity.py --run {i} --p32 {search_runs[i,0]} --min_length {search_runs[i,1]}"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    p.wait()

# gather summary parquet files
run_summaries = pd.concat(
    [pd.read_parquet(f) for f in Path(".").rglob("cluster_summary*.parquet")]
)

sns.set_style("whitegrid")

fig, ax = plt.subplots()
ax = sns.heatmap(
    run_summaries.pivot("p32", "min_length", "total_length"),
    cbar_kws={"label": "Total Connected Fracture Length"},
    cmap="viridis",
    square=True,
    fmt="g",
)
ax.invert_yaxis()
plt.yticks(rotation=0)
plt.ylabel("P32 (m$^{-1}$)")
plt.xlabel("Minimum Fracture Length (m)")
plt.show(block=False)

fig, ax = plt.subplots()
ax = sns.heatmap(
    run_summaries.pivot("p32", "min_length", "clusters"),
    cbar_kws={"label": "Number of Connected Clusters"},
    cmap="cividis",
    square=True,
    fmt="g",
)
ax.invert_yaxis()
plt.yticks(rotation=0)
plt.ylabel("P32 (m$^{-1}$)")
plt.xlabel("Minimum Fracture Length (m)")
plt.show(block=False)

fig, ax = plt.subplots()
ax = sns.heatmap(
    run_summaries.pivot("p32", "min_length", "num_lineaments"),
    cbar_kws={"label": "Number of Connected Lineaments"},
    cmap="cividis",
    square=True,
    fmt="g",
)
ax.invert_yaxis()
plt.yticks(rotation=0)
plt.ylabel("P32 (m$^{-1}$)")
plt.xlabel("Minimum Fracture Length (m)")
plt.show(block=False)

