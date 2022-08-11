# Notebook to make a power law plot and rose diagram for Azimuths
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import mplstereonet
from pyfracman.data import read_f2d_trace_file

## Duvernay
root_dir = Path("C:/repos/fracture_intensity_inference")
duvernay_path = Path("./case_studies/duvernay/generate_model")
seismic_faults = pd.read_csv(root_dir / duvernay_path / "seismic_fault_info.csv")[
    ["length", "strike", "dip"]
]
seismic_counts = pd.DataFrame(
    {
        "lengths": np.sort(seismic_faults.length.to_numpy()),
        "comp_count": np.arange(seismic_faults.shape[0], 0, -1),
        "Set": "Seismic",
    }
)

area_normalization = 400
nne_ssw_faults = read_f2d_trace_file(
    root_dir / duvernay_path / "NNE-SSW Traces.f2d"
).assign(Set="NNE-SSW")
nne_ssw_counts = pd.DataFrame(
    {
        "lengths": np.sort(nne_ssw_faults.length.to_numpy()),
        "comp_count": np.arange(nne_ssw_faults.shape[0], 0, -1) * area_normalization,
        "Set": "NNE-SSW",
    }
)

combined_counts = pd.concat([seismic_counts, nne_ssw_counts])

sns.set_style("whitegrid")
sns.set_palette("Set1")
f, ax = plt.subplots(figsize=(5, 5))

# bin lengths for power law plot
sns.scatterplot(data=combined_counts, x="lengths", y="comp_count", hue="Set", ax=ax)

ymax = combined_counts.comp_count.max()
xmin = 210
D = 1.8
X = np.linspace(200, 15000, 100)
sns.lineplot(x=X, y=(xmin / X) ** D * ymax, color="black", ax=ax)
ax.grid(b=True, which="major", color="black", linewidth=0.075)
ax.grid(b=True, which="minor", color="black", linewidth=0.075)
ax.text(2000, 400, f"Y = ({xmin}/X)^{1.8}", fontsize=10, weight="bold")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Normalized Complementary Cumulative Number")
plt.xlabel("Fracture Length (m)")
plt.show(block=False)

### Strike Rose Diagram
nnw_ssw_strikes = (nne_ssw_faults.strike).to_numpy()
seismic_strikes = (seismic_faults.strike).to_numpy()
bin_edges = np.arange(-5, 366, 10)

number_of_strikes, bin_edges = np.histogram(nnw_ssw_strikes, bin_edges)
number_of_strikes[0] += number_of_strikes[-1]
half = np.sum(np.split(number_of_strikes[:-1], 2), 0) * 2
nne_ssw_two_halves = np.concatenate([half, half])

number_of_strikes, bin_edges = np.histogram(seismic_strikes, bin_edges)
number_of_strikes[0] += number_of_strikes[-1]
half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
seismic_two_halves = np.concatenate([half, half])

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="polar")
ax.bar(
    np.deg2rad(np.arange(0, 360, 10)),
    seismic_two_halves,
    width=np.deg2rad(10),
    bottom=0.0,
    color="red",
    edgecolor="k",
)
ax.bar(
    np.deg2rad(np.arange(0, 360, 10)),
    nne_ssw_two_halves,
    width=np.deg2rad(10),
    bottom=0.0,
    color="blue",
    edgecolor="k",
)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetagrids(np.arange(0, 360, 15), labels=np.arange(0, 360, 15))
ax.set_rgrids(np.arange(1, seismic_two_halves.max(), 5), angle=0)
plt.show()


montney_path = Path("./case_studies/montney/generate_model")
filepath = root_dir / montney_path / "SeismicTraces.f2d"
is_faults = read_f2d_trace_file(root_dir / montney_path / "SeismicTraces.f2d").assign(
    Set="NE-SW"
)
is_counts = pd.DataFrame(
    {
        "lengths": np.sort(is_faults.length.to_numpy()),
        "comp_count": np.arange(is_faults.shape[0], 0, -1),
        "Set": "NE-SW",
    }
)

sns.set_style("whitegrid")
sns.set_palette("tab10")
f, ax = plt.subplots(figsize=(5, 5))

# bin lengths for power law plot
sns.scatterplot(data=is_counts, x="lengths", y="comp_count", hue="Set", ax=ax)


ymax = is_counts.comp_count.max()
xmin = 100
D = 1.01
X = np.linspace(xmin, is_counts.lengths.max(), 100)
sns.lineplot(x=X, y=(xmin / X) ** D * ymax, color="black", ax=ax)
ax.grid(b=True, which="major", color="black", linewidth=0.075)
ax.grid(b=True, which="minor", color="black", linewidth=0.075)

ax.text(400, 12, f"Y = ({xmin}/X)^{D}", fontsize=10, weight="bold")
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Complementary Cumulative Number")
plt.xlabel("Fracture Length (m)")
plt.show()


### Strike Rose Diagram
is_strikes = (is_faults.strike).to_numpy() - 90
regional_strikes = (
    pd.read_csv(root_dir / montney_path / "RegionalStrikes.csv").iloc[:, 0].to_numpy()
)
bin_edges = np.arange(-5, 366, 10)

number_of_strikes, bin_edges = np.histogram(is_strikes, bin_edges)
number_of_strikes[0] += number_of_strikes[-1]
half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
is_two_halves = np.concatenate([half, half])

number_of_strikes, bin_edges = np.histogram(regional_strikes, bin_edges)
number_of_strikes[0] += number_of_strikes[-1]
half = np.sum(np.split(number_of_strikes[:-1], 2), 0) * 2
regional_two_halves = np.concatenate([half, half])

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="polar")
ax.bar(
    np.deg2rad(np.arange(0, 360, 10)),
    regional_two_halves,
    width=np.deg2rad(10),
    bottom=0.0,
    color="red",
    edgecolor="k",
)
ax.bar(
    np.deg2rad(np.arange(0, 360, 10)),
    is_two_halves,
    width=np.deg2rad(10),
    bottom=0.0,
    color="blue",
    edgecolor="k",
)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetagrids(np.arange(0, 360, 15), labels=np.arange(0, 360, 15))
ax.set_rgrids(np.arange(1, seismic_two_halves.max(), 5), angle=0)
plt.show()
