# carbon-paradox

Carbon-aware grid forecasting project for DSAN 5550 focused on whether a more complex machine learning model can improve carbon intensity prediction enough to justify its own computational emissions.

The project uses hourly electricity generation by fuel from the U.S. Energy Information Administration (EIA) Grid Monitor API, along with eGRID reference data from the U.S. Environmental Protection Agency (EPA). These data are used to estimate regional carbon intensity, train forecasting models, and approximate regional graph connections for the GNN.

## Run

```powershell
uv sync --extra notebooks                    # install project and notebook dependencies
uv run python src/ingestion.py              # pull raw data and write processed source tables
# run notebooks/preprocess.ipynb            # build the model-ready forecasting dataset
uv run python src/baseline_model.py         # train the XGBoost baseline and save predictions
uv run python src/gnn_model.py              # train the graph neural network and save predictions
uv run python src/evaluation.py             # compare model performance and carbon tradeoffs
# run notebooks/carbon_paradox.ipynb        # view the final narrative analysis and results
```

