from pathlib import Path
from codecarbon import EmissionsTracker
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv

DATA_FILE = Path("data/processed/model_ready_data.csv")
EDGES_FILE = Path("data/processed/grid_edges.csv")
OUT_DIR = Path("out")
EIA_TO_EGRID = {
    "CISO": ["CAMX"],
    "ERCO": ["ERCT"],
    "ISNE": ["NEWE"],
    "MISO": ["MROE", "MROW"],
    "NYIS": ["NYCW", "NYLI", "NYUP"],
    "PJM": ["RFCE", "RFCW"],
    "SWPP": ["SPSO", "SRMW"],
}

def build_edge_index(regions, edges):
    pairs = []
    edge_pairs = set()

    for _, row in edges.iterrows():
        edge_pairs.add((row["source_region"], row["target_region"]))
        edge_pairs.add((row["target_region"], row["source_region"]))

    for i, region_i in enumerate(regions):
        pairs.append([i, i])

        for j, region_j in enumerate(regions):
            if i == j:
                continue

            left = EIA_TO_EGRID.get(region_i, [])
            right = EIA_TO_EGRID.get(region_j, [])

            connected = any((a, b) in edge_pairs for a in left for b in right)
            if connected:
                pairs.append([i, j])

    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def get_snapshot_data(df, timestamp, feature_columns, target_column, edges):
    snapshot = df[df["timestamp"] == timestamp].copy()
    snapshot = snapshot.dropna(subset=feature_columns + [target_column])
    snapshot = snapshot.sort_values("region").reset_index(drop=True)

    if len(snapshot) < 2:
        return None, None, None, None

    x = torch.tensor(snapshot[feature_columns].values, dtype=torch.float32)
    y = torch.tensor(snapshot[target_column].values, dtype=torch.float32)
    edge_index = build_edge_index(snapshot["region"].tolist(), edges)
    return snapshot, x, y, edge_index


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="carbon_paradox_gnn",
        output_dir=str(OUT_DIR),
        output_file="gnn_codecarbon.csv",
    )
    tracker.start()

    df = pd.read_csv(DATA_FILE)
    edges = pd.read_csv(EDGES_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["timestamp", "region"]).reset_index(drop=True)

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

    timestamps = sorted(df["timestamp"].unique())
    split_index = int(len(timestamps) * 0.8)
    train_times = timestamps[:split_index]
    test_times = timestamps[split_index:]

    train_df = df[df["timestamp"].isin(train_times)].copy()
    test_df = df[df["timestamp"].isin(test_times)].copy()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    train_df[feature_columns] = feature_scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = feature_scaler.transform(test_df[feature_columns])

    train_df[[target_column]] = target_scaler.fit_transform(train_df[[target_column]])
    test_df[[target_column]] = target_scaler.transform(test_df[[target_column]])

    conv1 = GCNConv(len(feature_columns), 16)
    conv2 = GCNConv(16, 1)
    optimizer = torch.optim.Adam(list(conv1.parameters()) + list(conv2.parameters()), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for _ in range(100):
        for timestamp in train_times:
            snapshot, x, y, edge_index = get_snapshot_data(
                train_df,
                timestamp,
                feature_columns,
                target_column,
                edges,
            )
            if snapshot is None:
                continue

            optimizer.zero_grad()
            predictions = conv1(x, edge_index)
            predictions = torch.relu(predictions)
            predictions = conv2(predictions, edge_index).squeeze()
            loss = loss_fn(predictions, y)
            loss.backward()
            optimizer.step()

    prediction_rows = []

    with torch.no_grad():
        for timestamp in test_times:
            snapshot, x, y, edge_index = get_snapshot_data(
                test_df,
                timestamp,
                feature_columns,
                target_column,
                edges,
            )
            if snapshot is None:
                continue

            predictions = conv1(x, edge_index)
            predictions = torch.relu(predictions)
            predictions = conv2(predictions, edge_index).squeeze().cpu().numpy()
            actual = target_scaler.inverse_transform(y.reshape(-1, 1).cpu().numpy()).flatten()
            predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

            out = snapshot[["timestamp", "region", target_column]].copy()
            out["actual"] = actual
            out["prediction"] = predictions
            out = out.drop(columns=[target_column])
            prediction_rows.append(out)

    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    mae = mean_absolute_error(predictions_df["actual"], predictions_df["prediction"])
    rmse = mean_squared_error(predictions_df["actual"], predictions_df["prediction"]) ** 0.5

    metrics = pd.DataFrame(
        {
            "metric": ["mae", "rmse"],
            "value": [mae, rmse],
        }
    )

    predictions_df.to_csv(OUT_DIR / "gnn_predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "gnn_metrics.csv", index=False)
    emissions = tracker.stop()

    print("GNN model complete")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Emissions (kg CO2eq): {emissions:.6f}")


if __name__ == "__main__":
    main()
