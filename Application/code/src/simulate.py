import pandas as pd
from data_gen import data_gen
from dml import dml_plr
from naive import naive_plr
from tqdm import tqdm

N_SIMS = 500
N_SAMPLES = 500
N_VARS = 20
N_SPLITS = 5
THETA_GT = 0.5
SEED = 67


def main():
    """Simulation runs for navi, DML (no cross-fitting) and DML Cross-Fitting"""
    results = []

    for i in tqdm(range(N_SIMS)):
        running_seed = SEED + i
        X, y, d = data_gen(N_SAMPLES, N_VARS, THETA_GT, running_seed)

        naive = naive_plr(X, y, d, running_seed)
        dml_of = dml_plr(X, y, d, N_SPLITS, False, running_seed)
        dml_cf = dml_plr(X, y, d, N_SPLITS, True, running_seed)

        results.append({
            "iter": i,
            "theta_naive": naive,
            "theta_dml_of": dml_of,
            "theta_dml_cf": dml_cf,
        })

    pd.DataFrame(results).to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()
