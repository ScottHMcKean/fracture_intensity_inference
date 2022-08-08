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

    # RUN AND MONITOR
    FracmanRunner().Run(macro_filepath)

    # POSTPROCESS
    # read scenario .fab file to get seismic length
    fab_info = parse_fab_file("SeismogenicFracs.fab")
    total_length = fab_info["property_df"].FractureLength.sum()
    num_lineaments = fab_info["property_df"].shape[0]
    print(str(total_length))
    print(str(num_lineaments))

    # read simulated .ors file for microseismic events
    # compare against stage centres
    simulated_events = read_ors_file("SubsampledEvents.ors")

    stages = pd.concat(
        [
            read_ors_file(f)
            for f in Path("../generate_model/").glob("Well*Midpoints.ors")
        ]
    )
    event_xy = simulated_events[["X[m]", "Y[m]"]].values
    stage_xy = stages[["X[m]", "Y[m]"]].values
    dist2well = cdist(event_xy, stage_xy).min(axis=1)

    # get quantiles
    d2w_quantiles = np.quantile(dist2well, [0.01, 0.05, 0.15, 0.50, 0.85, 0.95, 0.99])

    # bin simulated events into probability mass function with spec'd bins
    hist = np.histogram(
        dist2well,
        bins=np.array([0, 250, 500, 750]),
        density=True,
    )
    bin_width = hist[1][1:] - hist[1][:-1]  # max of bins limits
    prob_mass = hist[0] * bin_width  # values in bin

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
