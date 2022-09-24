from pyfracman.fab import parse_fab_file
from pyfracman.point_analysis import read_ors_file
from pyfracman.data import read_f2d_trace_file
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt

root_dir = Path("C:/repos/fracture_intensity_inference")
case_path = Path("./case_studies/duvernay")
set_name = "NNE_SSW"
fault_file = "NNE-SSW Traces.f2d"
event_file = "InducedMicroseismicity.csv"
xmin = 210
D = 1.8

# root_dir = Path("C:/repos/fracture_intensity_inference")
# case_path = Path("./case_studies/montney")
# set_name = "NE_SW"
# fault_file = "SeismicTraces.f2d"
# event_file = "InducedSeismicity.csv"
# xmin = 100
# D = 1.01

color_values = []
color_list = sns.color_palette("Set1")
for color in color_list:
    color_values.append([value * 255 for value in color])

faults = read_f2d_trace_file(root_dir / case_path / "generate_model" / fault_file)

counts = pd.DataFrame(
    {
        "lengths": np.sort(faults.length.to_numpy()),
        "comp_count": np.arange(faults.shape[0], 0, -1),
        "Set": set_name,
    }
)

# POSTPROCESS
# read scenario .fab file to get seismic length
fab_info = parse_fab_file(root_dir / case_path / "run_pest" / "SeismogenicFracs_1.fab")
total_length = fab_info["property_df"].FractureLength.sum()
num_lineaments = fab_info["property_df"].shape[0]

simulated_counts = pd.DataFrame(
    {
        "lengths": np.sort(fab_info["property_df"].FractureLength.to_numpy()),
        "comp_count": np.arange(fab_info["property_df"].shape[0], 0, -1),
        "Set": "Simulated",
    }
)

combined_counts = pd.concat([simulated_counts, counts])

sns.set_style("whitegrid")
sns.set_palette("Set1")

f, ax = plt.subplots(figsize=(5, 5))

# bin lengths for power law plot
sns.scatterplot(data=combined_counts, x="lengths", y="comp_count", hue="Set", ax=ax)

ymax = combined_counts.comp_count.max()
X = np.linspace(xmin * 0.9, simulated_counts.lengths.max() * 1.1, 100)
sns.lineplot(x=X, y=(xmin / X) ** D * ymax, color="black", ax=ax)
ax.grid(b=True, which="major", color="black", linewidth=0.075)
ax.grid(b=True, which="minor", color="black", linewidth=0.075)
ax.text(2000, 400, f"Y = ({xmin}/X)^{1.8}", fontsize=10, weight="bold")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Complementary Cumulative Number")
plt.xlabel("Fracture Length (m)")
plt.show(block=False)

# read simulated .ors file for microseismic events
# compare against stage centres

simulated_events = read_ors_file(
    root_dir / case_path / "run_pest" / "SubsampledEvents.ors"
)
simulated_event_xy = simulated_events[["X[m]", "Y[m]"]].values

actual_events = pd.read_csv(root_dir / case_path / "generate_model" / event_file)
actual_event_xy = actual_events[["X[m]", "Y[m]"]].values

stages = pd.concat(
    [
        read_ors_file(f)
        for f in (root_dir / case_path / "generate_model").glob("Well*Midpoints.ors")
    ]
)
stage_xy = stages[["X[m]", "Y[m]"]].values

simulated = pd.DataFrame(
    {
        "dist": np.sort(cdist(simulated_event_xy, stage_xy).min(axis=1)),
        "freq": np.array(range(simulated_event_xy.shape[0]))
        / float(simulated_event_xy.shape[0]),
        "Set": "Simulated",
    }
)

actual = pd.DataFrame(
    {
        "dist": np.sort(cdist(actual_event_xy, stage_xy).min(axis=1)),
        "freq": np.array(range(actual_event_xy.shape[0]))
        / float(actual_event_xy.shape[0]),
        "Set": set_name,
    }
)

combined_dist = pd.concat([simulated, actual])

sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=combined_dist, x="dist", y="freq", hue="Set", ax=ax, s=5)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
plt.ylabel("Cumulative Frequency")
plt.xlabel("Event Distance to Closest Stage (m)")
plt.show(block=False)
