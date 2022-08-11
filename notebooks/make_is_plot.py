# Notebook to make a power law plot and rose diagram for Azimuths
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyfracman.data import read_ors
import re

root_dir = Path("C:/repos/fracture_intensity_inference")
duvernay_path = Path("./case_studies/duvernay/generate_model")
montney_path = Path("./case_studies/montney/generate_model")
mag_sizes = (1, 50)
mag_size_norms = (-1, 3.1)

# DUVERNAY

is_data = pd.read_csv(root_dir / duvernay_path / "InducedMicroseismicity.csv").assign(
    moment=lambda x: np.power(10, x.Magnitude)
)
stages = pd.concat(
    [
        read_ors(well_filepath).assign(
            Well=well_filepath.name.replace(" Midpoints.ors", "")
        )
        for well_filepath in (root_dir / duvernay_path).rglob("Well*Midpoints.ors")
    ]
)

well_paths = []
for f in (root_dir / duvernay_path).rglob("Well*.txt"):
    if re.search(r"Well [A-Z|0-9].txt", f.name):
        print(f)
        txt_df = pd.read_table(f, header=None)
        data_start = txt_df[0].str.contains("#=========").idxmax()
        well_path = txt_df.iloc[data_start + 3 :, 0].str.split(expand=True)
        well_path.columns = ["MD", "X", "Y", "Z"]
        well_paths.append(
            well_path.apply(pd.to_numeric, axis=1).assign(
                Well=f.name.replace(".txt", "")
            )
        )

well_paths = pd.concat(well_paths).sort_values(["Well", "MD"]).reset_index(drop=True)

sns.set_style("whitegrid")
sns.set_palette("Set1")

fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=stages, x="x", y="y", hue="Well", size=10, ax=ax, legend=False)
sns.lineplot(data=well_paths, x="X", y="Y", hue="Well", ax=ax, sort=False, legend=False)
sns.scatterplot(
    data=is_data,
    x="X[m]",
    y="Y[m]",
    size="Magnitude",
    ax=ax,
    color="black",
    sizes=mag_sizes,
    size_norm=mag_size_norms,
)
ax.set_ylabel("Northing (m)")
ax.set_xlabel("Easting (m)")
plt.legend(title="Magnitude", bbox_to_anchor=(1.05, 0.99))
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show(block=False)

# MONTNEY

is_data = pd.read_csv(root_dir / montney_path / "InducedSeismicity.csv").assign(
    moment=lambda x: np.power(10, x.Magnitude)
)
stages = pd.concat(
    [
        read_ors(well_filepath).assign(
            Well=well_filepath.name.replace(" Midpoints.ors", "")
        )
        for well_filepath in (root_dir / montney_path).rglob("Well*Midpoints.ors")
    ]
)

well_paths = []
for f in (root_dir / montney_path).rglob("Well*.txt"):
    if re.search(r"Well [A-Z|0-9].txt", f.name):
        print(f)
        txt_df = pd.read_table(f, header=None)
        data_start = txt_df[0].str.contains("#=========").idxmax()
        well_path = txt_df.iloc[data_start + 3 :, 0].str.split(expand=True)
        well_path.columns = ["MD", "X", "Y", "Z"]
        well_paths.append(
            well_path.apply(pd.to_numeric, axis=1).assign(
                Well=f.name.replace(".txt", "")
            )
        )

well_paths = pd.concat(well_paths).sort_values(["Well", "MD"]).reset_index(drop=True)

well_paths[0].to_csv("test.csv")

sns.set_style("whitegrid")
sns.set_palette("Set1")

fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=stages, x="x", y="y", hue="Well", size=10, ax=ax, legend=False)
sns.lineplot(data=well_paths, x="X", y="Y", hue="Well", ax=ax, sort=False, legend=False)
sns.scatterplot(
    data=is_data,
    x="X[m]",
    y="Y[m]",
    size="Magnitude",
    color="black",
    ax=ax,
    sizes=mag_sizes,
    size_norm=mag_size_norms,
    legend=None,
)
ax.set_ylabel("Northing (m)")
ax.set_xlabel("Easting (m)")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show(block=False)
