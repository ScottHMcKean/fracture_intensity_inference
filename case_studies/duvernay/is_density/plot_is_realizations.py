# Plot IS Realizations
import pandas as pd
from pathlib import Path
from pyfracman.data import read_ors
import seaborn as sns
import matplotlib.pyplot as plt
import re

root_path = Path(r"C:\repos\fracture_intensity_inference\case_studies\duvernay")

# load wells
well_paths = []
for f in (root_path / "generate_model").rglob("Well*.txt"):
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

event_dfs = []
for event_file in (root_path / "is_density").rglob("SubsampledEvents_*.ors"):
    print(event_file.name)
    run = int(event_file.name.replace(".ors", "").replace("SubsampledEvents_", ""))
    events = read_ors(root_path / "is_density" / event_file.name).assign(run=run)
    event_dfs.append(events)

combined_events = pd.concat(event_dfs)

sns.set_style("whitegrid")
sns.set_palette("Set1")

level_bins = [0.05, 0.10, 0.20, 0.30, 0.40, 0.75, 0.90, 0.95]

fig, ax = plt.subplots(figsize=(5, 5))
sns.lineplot(data=well_paths, x="X", y="Y", hue="Well", ax=ax, sort=False, legend=False)
sns.kdeplot(
    data=combined_events.sample(5000).reset_index(),
    x="x",
    y="y",
    fill=True,
    cbar=True,
    ax=ax,
    common_norm=True,
    levels=level_bins,
    cmap="viridis",
    grid=1000,
)
sns.scatterplot(data=combined_events, x="x", y="y", s=1, color="black", alpha=0.15)
ax.set_ylabel("Northing (m)")
ax.set_xlabel("Easting (m)")
plt.gca().set_aspect("equal")
fig.axes[-1].set_yticklabels([f"{t:.2f}" for t in level_bins])
plt.tight_layout()
plt.show(block=False)
