from pathlib import Path
import pandas as pd
import requests
import shapefile

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def get_generation_data(api_key, start, end, regions):
    rows = []

    for region in regions:
        url = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "start": start,
            "end": end,
            "facets[respondent][]": region,
            "length": 5000,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }

        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()["response"]["data"]
        rows.extend(data)

    df = pd.DataFrame(rows)
    df = df[
        ["period", "respondent", "respondent-name", "type-name", "value"]
    ].copy()
    df.columns = [
        "timestamp",
        "region",
        "region_name",
        "fuel_type",
        "generation_mwh",
    ]
    return df


def get_emissions_table():
    data = [
        {"fuel_type": "Coal", "kgco2e_per_mwh": 1000.0},
        {"fuel_type": "Natural Gas", "kgco2e_per_mwh": 450.0},
        {"fuel_type": "Petroleum", "kgco2e_per_mwh": 780.0},
        {"fuel_type": "Other Gases", "kgco2e_per_mwh": 600.0},
        {"fuel_type": "Nuclear", "kgco2e_per_mwh": 12.0},
        {"fuel_type": "Hydro", "kgco2e_per_mwh": 24.0},
        {"fuel_type": "Wind", "kgco2e_per_mwh": 11.0},
        {"fuel_type": "Solar", "kgco2e_per_mwh": 45.0},
        {"fuel_type": "Biomass", "kgco2e_per_mwh": 230.0},
        {"fuel_type": "Geothermal", "kgco2e_per_mwh": 38.0},
    ]
    return pd.DataFrame(data)


def pick_column(columns, options):
    for option in options:
        for column in columns:
            if option.lower() in str(column).lower():
                return column
    return columns[0]


def get_egrid_region_table(excel_file):
    df = pd.read_excel(excel_file, sheet_name="SRL23")
    code_col = pick_column(df.columns, ["subrgn", "srcode", "region"])
    name_col = pick_column(df.columns, ["subrgnname", "srname", "name"])
    rate_col = pick_column(df.columns, ["co2", "rate"])

    out = df[[code_col, name_col, rate_col]].copy()
    out.columns = ["region_code", "region_name", "annual_co2_rate"]
    return out.drop_duplicates()


def get_subregion_table(shapefile_path):
    reader = shapefile.Reader(str(shapefile_path))
    records = [record.as_dict() for record in reader.iterRecords()]
    df = pd.DataFrame(records)

    code_col = pick_column(df.columns, ["subrgn", "srcode", "name"])
    name_col = pick_column(df.columns, ["subrgnname", "srname", "name"])

    out = df[[code_col, name_col]].copy()
    out.columns = ["region_code", "region_name"]
    return out.drop_duplicates()


def get_edges_table(shapefile_path):
    reader = shapefile.Reader(str(shapefile_path))
    records = [record.as_dict() for record in reader.iterRecords()]
    shapes = list(reader.iterShapes())

    df = pd.DataFrame(records)
    code_col = pick_column(df.columns, ["subrgn", "srcode", "name"])

    regions = df[code_col].astype(str).tolist()
    boxes = [shape.bbox for shape in shapes]
    edges = []
    tolerance = 0.05

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            left = boxes[i]
            right = boxes[j]

            overlaps_x = left[0] <= right[2] + tolerance and left[2] + tolerance >= right[0]
            overlaps_y = left[1] <= right[3] + tolerance and left[3] + tolerance >= right[1]

            if overlaps_x and overlaps_y:
                edges.append(
                    {
                        "source_region": regions[i],
                        "target_region": regions[j],
                        "edge_type": "adjacent_subregion",
                    }
                )

    return pd.DataFrame(edges).drop_duplicates()


def save_tables(api_key, start, end, regions, excel_file, shapefile_path):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    generation = get_generation_data(api_key, start, end, regions)
    emissions = get_emissions_table()
    region_reference = get_egrid_region_table(excel_file)
    subregions = get_subregion_table(shapefile_path)
    edges = get_edges_table(shapefile_path)

    generation.to_csv(PROCESSED_DIR / "generation_by_fuel_hourly.csv", index=False)
    emissions.to_csv(PROCESSED_DIR / "emissions_factors_by_fuel.csv", index=False)
    region_reference.to_csv(PROCESSED_DIR / "egrid_region_reference.csv", index=False)
    subregions.to_csv(PROCESSED_DIR / "egrid_subregions.csv", index=False)
    edges.to_csv(PROCESSED_DIR / "grid_edges.csv", index=False)


if __name__ == "__main__":
    API_KEY = "eLTazpiHlIzXXf1c2iolPzhgsoPzhSxbUrqsqxhi"
    START = "2024-01-01T00"
    END = "2024-01-07T23"
    REGIONS = ["PJM", "NYIS", "CISO"]

    EXCEL_FILE = RAW_DIR / "egrid2023_data_rev2.xlsx"
    SHAPEFILE_PATH = RAW_DIR / "eGRID2023_Subregions.shp"

    save_tables(API_KEY, START, END, REGIONS, EXCEL_FILE, SHAPEFILE_PATH)
