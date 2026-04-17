import json
from pathlib import Path


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


path = Path("ABC_SMC2.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

analysis_cells = [
    md_cell(
        """## ABC-SMC Run

Run your `abc_smc(...)` function in the next code cell or paste your existing implementation here.
The analysis section below assumes the run returns:

- `populations`: a list of dictionaries with keys `particles`, `weights`, `distances`, `epsilon`
- `all_distances`: a list of accepted distance arrays for each round

If you already executed the SMC run in another session, just rerun that cell before the analysis cells below.
"""
    ),
    code_cell(
        """# Example placeholder:
# rng = np.random.default_rng(123)
# sigma = {k: pd.read_csv("pilot_results.csv")[f"s_{k}"].std() for k in STAT_NAMES}
# eps_schedule = [1.8, 1.4, 1.1, 0.9]
# populations, all_distances = abc_smc(
#     n_particles=100,
#     eps_schedule=eps_schedule,
#     n_reps=REPS_PER_DRAW,
#     sigma=sigma,
#     rng=rng,
# )
"""
    ),
    md_cell(
        """## Analysis

The cells below turn the final ABC-SMC population into a posterior summary and compare it with the rejection ABC baselines.
"""
    ),
    code_cell(
        """pilot = pd.read_csv("pilot_results.csv")
sigma = {k: pilot[f"s_{k}"].std() for k in STAT_NAMES}

if "populations" not in globals():
    raise RuntimeError("Run the ABC-SMC sampler first so that `populations` exists.")

final_population = populations[-1]

smc_df = pd.DataFrame(final_population["particles"], columns=["beta", "gamma", "rho"])
smc_df["weight"] = np.asarray(final_population["weights"])
smc_df["distance"] = np.asarray(final_population["distances"])

smc_df.head()
"""
    ),
    code_cell(
        """def weighted_quantile(values, quantiles, sample_weight):
    values = np.asarray(values)
    quantiles = np.asarray(quantiles)
    sample_weight = np.asarray(sample_weight)

    order = np.argsort(values)
    values = values[order]
    sample_weight = sample_weight[order]

    cumulative = np.cumsum(sample_weight) / sample_weight.sum()
    return np.interp(quantiles, cumulative, values)


def weighted_summary(df, params=("beta", "gamma", "rho"), weight_col="weight"):
    w = df[weight_col].to_numpy()
    w = w / w.sum()
    rows = []
    for param in params:
        vals = df[param].to_numpy()
        mean = np.average(vals, weights=w)
        var = np.average((vals - mean) ** 2, weights=w)
        q025, q50, q975 = weighted_quantile(vals, [0.025, 0.5, 0.975], w)
        rows.append(
            {
                "parameter": param,
                "mean": mean,
                "sd": np.sqrt(var),
                "median": q50,
                "q025": q025,
                "q975": q975,
            }
        )
    return pd.DataFrame(rows)


smc_summary = weighted_summary(smc_df)
smc_summary
"""
    ),
    code_cell(
        """round_rows = []
for i, pop in enumerate(populations, start=1):
    w = np.asarray(pop["weights"])
    w = w / w.sum()
    distances = np.asarray(pop["distances"])
    ess = 1.0 / np.sum(w ** 2)
    round_rows.append(
        {
            "round": i,
            "epsilon": pop["epsilon"],
            "n_particles": len(pop["particles"]),
            "mean_distance": np.average(distances, weights=w),
            "median_distance": weighted_quantile(distances, [0.5], w)[0],
            "min_distance": distances.min(),
            "max_distance": distances.max(),
            "ess": ess,
        }
    )

round_df = pd.DataFrame(round_rows)
round_df
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(round_df["round"], round_df["epsilon"], marker="o")
axes[0].set_title("Tolerance schedule")
axes[0].set_xlabel("Round")
axes[0].set_ylabel("epsilon")

axes[1].plot(round_df["round"], round_df["mean_distance"], marker="o", label="weighted mean")
axes[1].plot(round_df["round"], round_df["median_distance"], marker="s", label="weighted median")
axes[1].set_title("Accepted distances by round")
axes[1].set_xlabel("Round")
axes[1].set_ylabel("distance")
axes[1].legend()

axes[2].plot(round_df["round"], round_df["ess"], marker="o")
axes[2].set_title("Effective sample size")
axes[2].set_xlabel("Round")
axes[2].set_ylabel("ESS")

plt.tight_layout()
plt.show()
"""
    ),
    code_cell(
        """fig, axes = plt.subplots(1, 3, figsize=(15, 4))

weights = smc_df["weight"].to_numpy()
for ax, param in zip(axes, ["beta", "gamma", "rho"]):
    ax.hist(smc_df[param], bins=25, weights=weights, density=True, alpha=0.75, color="#2F6B8A")
    ax.set_title(f"ABC-SMC posterior: {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("density")

plt.tight_layout()
plt.show()
"""
    ),
    code_cell(
        """rejection_confirm = pd.read_csv("abc_results_confirm.csv")
rejection_full = pd.read_csv("abc_results_full.csv")

comparison_rows = []
for param in ["beta", "gamma", "rho"]:
    smc_vals = smc_df[param].to_numpy()
    smc_w = smc_df["weight"].to_numpy()
    smc_mean = np.average(smc_vals, weights=smc_w)
    smc_sd = np.sqrt(np.average((smc_vals - smc_mean) ** 2, weights=smc_w))
    smc_q025, smc_q975 = weighted_quantile(smc_vals, [0.025, 0.975], smc_w)

    comparison_rows.append(
        {
            "parameter": param,
            "smc_mean": smc_mean,
            "smc_sd": smc_sd,
            "smc_q025": smc_q025,
            "smc_q975": smc_q975,
            "rej5_mean": rejection_confirm[param].mean(),
            "rej5_sd": rejection_confirm[param].std(),
            "rej1_mean": rejection_full[param].mean(),
            "rej1_sd": rejection_full[param].std(),
        }
    )

comparison_df = pd.DataFrame(comparison_rows)
comparison_df
"""
    ),
    code_cell(
        """pairs = [("beta", "gamma"), ("beta", "rho"), ("gamma", "rho")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (x_name, y_name) in zip(axes, pairs):
    ax.scatter(rejection_full[x_name], rejection_full[y_name], s=16, alpha=0.25, label="Rejection ABC (1%)")
    ax.scatter(
        smc_df[x_name],
        smc_df[y_name],
        s=40 + 200 * smc_df["weight"],
        alpha=0.7,
        label="ABC-SMC",
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{x_name} vs {y_name}")
    ax.legend()

plt.tight_layout()
plt.show()
"""
    ),
    code_cell(
        """obs_summary = pd.Series(s_obs, name="observed")

def summarize_simulated_summaries(df, weight_col=None):
    summary_cols = [f"s_{k}" for k in STAT_NAMES if f"s_{k}" in df.columns]
    if len(summary_cols) == len(STAT_NAMES):
        if weight_col is None:
            values = {col[2:]: df[col].mean() for col in summary_cols}
        else:
            w = df[weight_col].to_numpy()
            values = {col[2:]: np.average(df[col], weights=w) for col in summary_cols}
        return pd.Series(values)
    return None


if all(f"s_{k}" in smc_df.columns for k in STAT_NAMES):
    smc_sim_summary = summarize_simulated_summaries(smc_df, weight_col="weight")
else:
    smc_sim_summary = None

rej5_summary = summarize_simulated_summaries(rejection_confirm)
rej1_summary = summarize_simulated_summaries(rejection_full)

summary_comparison = pd.DataFrame({"observed": obs_summary, "rej5": rej5_summary, "rej1": rej1_summary})
if smc_sim_summary is not None:
    summary_comparison["smc"] = smc_sim_summary

summary_comparison["rej1_minus_obs"] = summary_comparison["rej1"] - summary_comparison["observed"]
if smc_sim_summary is not None:
    summary_comparison["smc_minus_obs"] = summary_comparison["smc"] - summary_comparison["observed"]

summary_comparison
"""
    ),
    md_cell(
        """## Interpretation

Use the outputs above to make three points:

1. `round_df` should show the ABC-SMC populations getting more selective as epsilon decreases.
2. `comparison_df` shows whether the ABC-SMC posterior is similar to or tighter than the rejection ABC posterior.
3. The pairwise scatter plots are especially useful for the `beta-rho` tradeoff: if the SMC cloud is more concentrated along the same ridge, then SMC is refining the posterior even if the wall-clock runtime is longer.

A short write-up template:

“The ABC-SMC posterior concentrates progressively over the sequence of tolerances, with later populations achieving lower accepted distances to the observed summaries. Relative to vanilla rejection ABC, the final ABC-SMC population should be interpreted as a weighted posterior approximation rather than an equally weighted accepted sample. In this problem the main identifiability feature remains the `beta-rho` tradeoff, but ABC-SMC provides a more structured sequential refinement of that posterior region.” 
"""
    ),
]

nb["cells"] = nb["cells"][:8] + analysis_cells
path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print("Updated ABC_SMC2.ipynb")
