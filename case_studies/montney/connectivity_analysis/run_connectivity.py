# File to run connectivity analysis in a similar fashion to PEST
# Uses a .ptf file, but instead of PEST perturbing it, we do it in Python

from pyfracman.run import FracmanRunner
from pyfracman.fab import parse_fab_file
import argparse
import pandas as pd
import numpy as np

def run_connectivity_analysis(args):

    # read ptf file
    with open(args.file, "r") as f:
        fmf_out = (
            f.read()
            .replace("$ min_length $", str(args.min_length))
            .replace("$ p32 $", str(args.p32))
            .replace("$ run $", str(args.run))
        )

    with open(args.file.replace(".ptf", ".fmf"), "w") as f:
        f.write(fmf_out)

    FracmanRunner().Run(args.file.replace(".ptf", ".fmf"))

    # read cluster fractures and get total fracture length
    try:
        fab_info = parse_fab_file("cluster_fractures_" + str(args.run) + ".fab")
        length = fab_info["property_df"].FractureLength.sum()
        num_lineaments = fab_info["property_df"].shape[0]
        clusters = np.unique(fab_info["sets"]).shape[0]
    except Exception as e:
        print(e)
        length = 0
        num_lineaments = 0
        clusters = 0

    # write run summary
    pd.DataFrame(
        {
            "run": args.run,
            "p32": args.p32,
            "min_length": args.min_length,
            "total_length": length,
            "num_lineaments": num_lineaments,
            "clusters": clusters,
        },
        index=[args.run],
    ).to_parquet("cluster_summary_" + str(args.run) + ".parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="connectivity.ptf")
    parser.add_argument("--run", type=int, default=2, help="run integer code")
    parser.add_argument("--p32", type=float, default=1e-2)
    parser.add_argument("--min_length", type=float, default=50)
    args = parser.parse_args()
    run_connectivity_analysis(args)
