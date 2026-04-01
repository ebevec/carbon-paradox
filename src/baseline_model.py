from pathlib import Path
from codecarbon import EmissionsTracker
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_FILE = Path("data/processed/model_ready_data.csv")
OUT_DIR = Path("out")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(
        project_name="carbon_paradox_baseline",
        output_dir=str(OUT_DIR),
        output_file="baseline_codecarbon.csv",
    )
    tracker.start()

    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_columns = [
        "total_generation_mwh",
        "total_co2_kg",
        "carbon_intensity_kg_per_mwh",
        "lag_1",
        "lag_24",
        "hour",
        "day_of_week",
        "month",
    ]
    target_column = "target_carbon_intensity_24h"

    X = df[feature_columns]
    y = df[target_column]

    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    test_data = df.iloc[split_index:].copy()

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    metrics = pd.DataFrame(
        {
            "metric": ["mae", "rmse"],
            "value": [mae, rmse],
        }
    )

    test_data["actual"] = y_test.values
    test_data["prediction"] = predictions

    metrics.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)
    test_data.to_csv(OUT_DIR / "baseline_predictions.csv", index=False)
    emissions = tracker.stop()

    print("Baseline model complete")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Emissions (kg CO2eq): {emissions:.6f}")


if __name__ == "__main__":
    main()
