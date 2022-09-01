from pathlib import Path
import pandas as pd

import os

os.chdir(r"C:\repos\fracture_intensity_inference\case_studies\duvernay\run_pest")

sts_series = []
for sts_file in Path(".").glob("SeismogenicFaultIntensity_*.sts"):
    sts_raw = pd.read_csv(sts_file, sep="\t", header=None)
    sts_raw[0] = sts_raw[0].str.strip()
    sts_raw = sts_raw.drop(2, axis=1)
    sts_raw = sts_raw.set_index(0)
    sts_raw[1].transpose()
    sts_series.append(sts_raw)

sts_compiled = pd.concat(sts_series, axis=1).transpose()
