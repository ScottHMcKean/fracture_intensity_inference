import os

os.chdir(r"C:\repos\fracture_intensity_inference\case_studies\montney\run_pest")

from pyfracman.run import FracmanRunner
from pyfracman.fab import parse_fab_file
from pyfracman.point_analysis import read_ors_file
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def run_simulation(macro_filepath):

    # PREPROCESS
    for fab_file in Path(".").rglob("SeismogenicFracs_*.fab"):
        try:
            fab_file.unlink()
        except OSError as e:
            print(f"Error {fab_file.name}")

    for sim_event_file in Path(".").glob("SubsampledEvents_*.ors"):
        try:
            sim_event_file.unlink()
        except OSError as e:
            print(f"Error {sim_event_file.name}")

    # RUN AND MONITOR
    FracmanRunner().Run(macro_filepath)

    # POSTPROCESS
    # read scenario .fab file to get seismic length
    total_length_list = []
    num_lineaments_list = []
    for fab_file in Path(".").rglob("SeismogenicFracs_*.fab"):
        fab_info = parse_fab_file(fab_file)
        total_length = fab_info["property_df"].FractureLength.sum()
        num_lineaments = fab_info["property_df"].shape[0]
        total_length_list.append(total_length)
        num_lineaments_list.append(num_lineaments)

    total_length = np.mean(total_length_list)
    print(total_length)
    num_lineaments = np.mean(num_lineaments_list)
    print(num_lineaments)

    stages = pd.concat(
        [
            read_ors_file(f)
            for f in Path("../generate_model/").glob("Well*Midpoints.ors")
        ]
    )
    stage_xy = stages[["X[m]", "Y[m]"]].values

    dist2well_list = []
    d2w_quantiles_list = []
    prob_mass_list = []
    for sim_event_file in Path(".").glob("SubsampledEvents_*.ors"):
        simulated_events = read_ors_file(sim_event_file)
        event_xy = simulated_events[["X[m]", "Y[m]"]].values
        dist2well = cdist(event_xy, stage_xy).min(axis=1)
        dist2well_list.append(dist2well)
        # get quantiles
        d2w_quantiles = np.quantile(
            dist2well, [0.01, 0.05, 0.15, 0.50, 0.85, 0.95, 0.99]
        )
        d2w_quantiles_list.append(d2w_quantiles)

        # bin simulated events into probability mass function with spec'd bins
        hist = np.histogram(
            dist2well,
            bins=np.array([0, 250, 500, 750]),
            density=True,
        )
        bin_width = hist[1][1:] - hist[1][:-1]  # max of bins limits
        prob_mass = hist[0] * bin_width  # values in bin
        prob_mass_list.append(prob_mass)

    d2w_quantiles = np.vstack([np.array(i) for i in d2w_quantiles_list]).mean(axis=0)
    prob_mass = np.vstack([np.array(i) for i in prob_mass_list]).mean(axis=0)

    with open("output.sts", "w") as f:
        f.write(f"{total_length}\n")
        f.write(f"{num_lineaments}\n")
        for d in d2w_quantiles:
            f.write(f"{d}\n")
        for p in prob_mass:
            f.write(f"{p}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fracman macro runner")
    parser.add_argument("macro", type=str, nargs=1, help="FracMan macro filename")
    args = parser.parse_args()
    run_simulation(args.macro[0])
