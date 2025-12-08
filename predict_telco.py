import argparse
import pandas as pd
import joblib


def main():
    parser = argparse.ArgumentParser(description="Local Telco Churn Prediction Pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV with new customers")
    parser.add_argument("--output", default="predictions.csv", help="Where to save predictions")
    args = parser.parse_args()

    # Load pipeline
    print("Loading trained pipeline (sgd_pipeline.pkl)...")
    pipeline = joblib.load("sgd_pipeline.pkl")

    # Load new data
    print(f"Loading {args.input}...")
    df_new = pd.read_csv(args.input)

    # Drop ID column if present
    if "customerID" in df_new.columns:
        df_new = df_new.drop(columns=["customerID"])

    print("Generating churn probabilities...")
    proba = pipeline.predict_proba(df_new)[:, 1]

    df_new["Churn_Probability"] = proba
    df_new.to_csv(args.output, index=False)

    print(f"Saved predictions → {args.output}")


if __name__ == "__main__":
    main()
