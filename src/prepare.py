import argparse
import pandas as pd
import os


def clean_area_size(val):
    # Some rows have numeric in 'Area Size' (already numeric), others like '4 Marla'
    try:
        return float(val)
    except Exception:
        try:
            # extract first number from strings like '4 Marla' or '4.5 Marla'
            import re

            m = re.search(r"([0-9]+\.?[0-9]*)", str(val))
            if m:
                return float(m.group(1))
        except Exception:
            return None
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, default="data/zameen-updated.csv")
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.in_csv)
    # Basic cleaning: select columns useful for modeling
    cols_needed = [
        "price",
        "bedrooms",
        "baths",
        "Area Size",
        "property_type",
        "city",
        "latitude",
        "longitude",
    ]
    # keep only available columns
    cols_present = [c for c in cols_needed if c in df.columns]
    df = df[cols_present].copy()

    # normalize area size to numeric
    if "Area Size" in df.columns:
        df["Area Size"] = df["Area Size"].apply(clean_area_size)

    # drop rows missing target
    df = df[pd.to_numeric(df["price"], errors="coerce").notnull()]
    df["price"] = df["price"].astype(float)

    out_path = os.path.join(args.out_dir, "house_prices.csv")
    df.to_csv(out_path, index=False)
    print("Saved prepared house_prices.csv to", out_path)
