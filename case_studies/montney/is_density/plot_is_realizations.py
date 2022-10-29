# Plot IS Realizations
import pandas as pd
from pathlib import Path
from pyfracman.data import read_ors
import seaborn as sns
import matplotlib.pyplot as plt
import re
import matplotlib.ticker as ticker

root_path = Path(r"C:\repos\fracture_intensity_inference\case_studies\montney")

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

level_bins = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

fig, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(
    data=well_paths,
    x="X",
    y="Y",
    hue="Well",
    palette=["k", "k", "k"],
    ax=ax,
    sort=False,
    legend=False,
    linewidth=1,
)
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
)
sns.set_style({"axes.gridcolor": "#000000"})
ax.set_ylabel("")
ax.set_xlabel("")
plt.gca().set_aspect("equal")
ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
ax.set_ylim(-1500, 1000)
ax.set_xlim(-500, 2500)
fig.axes[-1].set_yticklabels([f"{t:.2f}" for t in level_bins])
plt.tight_layout()
plt.savefig("montney_is_realizations.pdf")
plt.savefig("montney_is_realizations.png")
plt.show(block=False)
