from pathlib import Path
import pandas as pd

OUT_DIR = Path("out")
LOAD_KWH = 10

def compare_models():
    baseline = pd.read_csv(OUT_DIR / "baseline_metrics.csv")
    gnn = pd.read_csv(OUT_DIR / "gnn_metrics.csv")

    comparison = baseline.merge(gnn, on="metric", suffixes=("_baseline", "_gnn"))
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    return comparison


def simulate_charging_shift(predictions_file, model_name):
    df = pd.read_csv(predictions_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    rows = []

    for (region, date), group in df.groupby(["region", "date"]):
        group = group.sort_values("timestamp").copy()

        random_actual = group["actual"].mean()
        best_row = group.loc[group["prediction"].idxmin()]
        optimized_actual = best_row["actual"]
        savings_kg = (LOAD_KWH / 1000) * (random_actual - optimized_actual)

        rows.append(
            {
                "model": model_name,
                "region": region,
                "date": date,
                "random_actual_carbon_intensity": random_actual,
                "optimized_actual_carbon_intensity": optimized_actual,
                "emissions_saved_kg": savings_kg,
            }
        )

    result = pd.DataFrame(rows)
    return result


def calculate_break_even(codecarbon_file, simulation_df, model_name):
    emissions = pd.read_csv(codecarbon_file)["emissions"].sum()
    average_savings = simulation_df["emissions_saved_kg"].mean()
    sessions_to_break_even = emissions / average_savings

    return pd.DataFrame(
        {
            "model": [model_name],
            "training_emissions_kg": [emissions],
            "average_savings_per_session_kg": [average_savings],
            "sessions_to_break_even": [sessions_to_break_even],
        }
    )


def main():
    comparison = compare_models()

    baseline_sim = simulate_charging_shift(
        OUT_DIR / "baseline_predictions.csv",
        "baseline",
    )
    gnn_sim = simulate_charging_shift(
        OUT_DIR / "gnn_predictions.csv",
        "gnn",
    )
    simulation = pd.concat([baseline_sim, gnn_sim], ignore_index=True)

    baseline_break_even = calculate_break_even(
        OUT_DIR / "baseline_codecarbon.csv",
        baseline_sim,
        "baseline",
    )
    gnn_break_even = calculate_break_even(
        OUT_DIR / "gnn_codecarbon.csv",
        gnn_sim,
        "gnn",
    )
    break_even = pd.concat([baseline_break_even, gnn_break_even], ignore_index=True)

    simulation.to_csv(OUT_DIR / "charging_simulation.csv", index=False)
    break_even.to_csv(OUT_DIR / "break_even_analysis.csv", index=False)

    print(comparison)
    print()
    print(break_even)


if __name__ == "__main__":
    main()
