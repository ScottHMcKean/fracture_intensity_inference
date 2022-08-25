# File to run connectivity analysis in a similar fashion to PEST
# Uses a .ptf file, but instead of PEST perturbing it, we do it in Python

from pyfracman.run import FracmanRunner

file = "duvernay_is_realization.ptf"

for run in range(1, 101):
    print(run)
    # read ptf file
    with open(file, "r") as f:
        fmf_out = f.read().replace("$ run $", str(run))

    with open(file.replace(".ptf", ".fmf"), "w") as f:
        f.write(fmf_out)

    FracmanRunner().Run(file.replace(".ptf", ".fmf"))
