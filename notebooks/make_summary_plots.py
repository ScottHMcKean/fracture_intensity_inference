from pyfracman.fab import parse_fab_file
from pyfracman.point_analysis import read_ors_file
from pyfracman.data import read_f2d_trace_file
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt

case = "duvernay"  # "montney" or "duvernay"

if case == "duvernay":
    root_dir = Path("C:/repos/fracture_intensity_inference")
    case_path = Path("./case_studies/duvernay")
    set_name = "NNE_SSW"
    fault_file = "NNE-SSW Traces.f2d"
    event_file = "InducedMicroseismicity.csv"
    xmin = 210
    D = 1.8

if case == "montney":
    root_dir = Path("C:/repos/fracture_intensity_inference")
    case_path = Path("./case_studies/montney")
    set_name = "NE_SW"
    fault_file = "SeismicTraces.f2d"
    event_file = "InducedSeismicity.csv"
    xmin = 100
    D = 1.01

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

simulations = []
for run in range(1, 21):
    fab_info = parse_fab_file(
        root_dir / case_path / "run_pest" / f"SeismogenicFracs_{run}.fab"
    )
    total_length = fab_info["property_df"].FractureLength.sum()
    num_lineaments = fab_info["property_df"].shape[0]

    simulated_counts = pd.DataFrame(
        {
            "lengths": np.sort(fab_info["property_df"].FractureLength.to_numpy()),
            "comp_count": np.arange(fab_info["property_df"].shape[0], 0, -1),
            "Set": "Simulated",
            "Run": run,
        }
    )

    simulations.append(simulated_counts)

combined_counts = pd.concat(simulations + [counts])

sns.color_palette("Set1").as_hex()

sns.set_style("whitegrid")
sns.set_palette("Set1")
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(
    data=combined_counts.query("Set == 'Simulated'"),
    x="lengths",
    y="comp_count",
    color="#e41a1c",
    s=10,
    label="Simulated",
    edgecolor="none",
    ax=ax,
)
sns.scatterplot(
    data=combined_counts.query(f"Set == '{set_name}'"),
    x="lengths",
    y="comp_count",
    color="#377eb8",
    label=set_name,
    edgecolor="none",
    ax=ax,
)
ymax = combined_counts.query(f"Set == '{set_name}'").comp_count.max()
X = np.linspace(xmin * 0.9, simulated_counts.lengths.max() * 1.1, 100)
sns.lineplot(x=X, y=(xmin / X) ** D * ymax, color="black", ax=ax)
ax.grid(b=True, which="major", color="black", linewidth=0.075)
ax.grid(b=True, which="minor", color="black", linewidth=0.075)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Complementary Cumulative Number")
plt.xlabel("Fracture Length (m)")
plt.savefig(f"{case}_cum_number.pdf")
plt.savefig(f"{case}_cum_number.png")
plt.show(block=False)

# read simulated .ors file for microseismic events
stages = pd.concat(
    [
        read_ors_file(f)
        for f in (root_dir / case_path / "generate_model").glob("Well*Midpoints.ors")
    ]
)
stage_xy = stages[["X[m]", "Y[m]"]].values

# compare against stage centres
actual_events = pd.read_csv(root_dir / case_path / "generate_model" / event_file)
actual_event_xy = actual_events[["X[m]", "Y[m]"]].values
actual = pd.DataFrame(
    {
        "dist": np.sort(cdist(actual_event_xy, stage_xy).min(axis=1)),
        "freq": np.array(range(actual_event_xy.shape[0]))
        / float(actual_event_xy.shape[0]),
        "Set": set_name,
    }
)

simulations = []
for run in range(1, 21):

    simulated_events = read_ors_file(
        root_dir / case_path / "run_pest" / f"SubsampledEvents_{run}.ors"
    )
    simulated_event_xy = simulated_events[["X[m]", "Y[m]"]].values

    simulated = pd.DataFrame(
        {
            "dist": np.sort(cdist(simulated_event_xy, stage_xy).min(axis=1)),
            "freq": np.array(range(simulated_event_xy.shape[0]))
            / float(simulated_event_xy.shape[0]),
            "Set": "Simulated",
        }
    )

    simulations.append(simulated)

combined_dist = pd.concat(simulations + [actual])

sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(
    data=combined_dist.query("Set == 'Simulated'"),
    x="dist",
    y="freq",
    color="#e41a1c",
    s=3,
    label="Simulated",
    edgecolor="none",
    ax=ax,
)
sns.scatterplot(
    data=combined_dist.query(f"Set == '{set_name}'"),
    x="dist",
    y="freq",
    color="#377eb8",
    s=4,
    label=set_name,
    edgecolor="none",
    ax=ax,
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
plt.ylabel("Cumulative Frequency")
plt.xlabel("Event Distance to Closest Stage (m)")
plt.savefig(f"{case}_cum_freq.pdf")
plt.savefig(f"{case}_cum_freq.png")
plt.show(block=False)
