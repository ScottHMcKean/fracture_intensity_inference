# File to parse a PEST .rec file to provide a graph of the optimization history
import pandas as pd

obs_groups = ["lineaments", "d2w"]
parameters = ["p32a", "p32b", "truncmin"]

rec = pd.read_csv("notebooks/test1.rec", header=None)

# get iteration indices
iteration_start = rec[rec[0].str.contains("OPTIMISATION ITERATION NO.")].index.to_list()
iteration_start.append(len(rec))

# pull an iteration
iteration_dfs = []
for it_idx in range(len(iteration_start) - 1):
    iteration = (
        rec[iteration_start[it_idx] : iteration_start[it_idx + 1]]
        .iloc[:, 0]
        .reset_index(drop=True)
    )

    # parse output for a dataframe
    iter_info = {}
    iter_info["iteration"] = int(iteration[0].split(":")[1])

    # get phi values
    for obs_group in obs_groups:
        iter_info[obs_group + "_phi"] = float(
            iteration[iteration.str.contains(obs_group)].iloc[0].split(":")[1]
        )

    try:
        iter_info["iter_phi"] = float(
            iteration[iteration.str.contains("Phi =")]
            .str.extract("Phi = (.+) \(")
            .iloc[0]
        )
    except Exception:
        iter_info["iter_phi"] = pd.NA

    try:
        # get parameter values
        for param in parameters:
            iter_info[param] = float(
                iteration[iteration.str.contains(param)].iloc[0].split()[1]
            )
    except Exception:
        for param in parameters:
            iter_info[param] = pd.NA

    iteration_dfs.append(pd.DataFrame(iter_info, index=[iter_info["iteration"]]))

pd.concat(iteration_dfs)
