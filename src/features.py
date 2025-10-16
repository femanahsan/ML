import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    # fill numeric missing values
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "biufc"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c].fillna(df[c].median(), inplace=True)

    # simple one-hot encoding for property_type and city
    cat_cols = [c for c in ["property_type", "city"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # target
    y = df["price"].values
    X = df.drop(columns=["price"]).values
    # ensure numeric float arrays (convert object dtypes if any)
    X = np.array(X, dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)
    print("Train/test data saved in", args.out_dir)
