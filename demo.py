
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

RESULTS = Path("results")

# --------------------------------------------------
# 1. LOAD DATA, FEATURES, AND MODELS
# --------------------------------------------------

data = pd.read_csv(
    RESULTS / "cleaned_data.csv",
    index_col=0,
    parse_dates=True
)

features = joblib.load(RESULTS / "model_features.pkl")

models = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

# Match the chronological test split used in training
test = data.loc[data.index >= "2023-01-01"]

# --------------------------------------------------
# 2. PREDICTION FUNCTION
# --------------------------------------------------

def show_predictions(X, target_date, actual=None):
    print(f"\nPredictions for {target_date}")
    print("-" * 50)

    for name, filename in models.items():
        model = joblib.load(RESULTS / filename)
        prediction = model.predict(X[features])[0]
        print(f"{name}: {prediction:.2f} °C")

    if actual is not None and pd.notna(actual):
        print(f"Actual next-day temperature: {actual:.2f} °C")


# --------------------------------------------------
# 3. SELECT DEMONSTRATION MODE
# --------------------------------------------------

print("\nCHENNAI TEMPERATURE PREDICTION DEMO")
print("=" * 50)
print("1. Historical test date")
print("2. Future-date simulation with manual inputs")

mode = input("\nChoose mode (1 or 2): ").strip()


# --------------------------------------------------
# MODE 1: HISTORICAL TEST-DATA PREDICTION
# --------------------------------------------------

if mode == "1":
    date_input = input(
        "Enter a test date (YYYY-MM-DD): "
    ).strip()

    try:
        date = pd.Timestamp(date_input)

        if date not in test.index:
            print(
                "\nDate unavailable in test data."
                f"\nAvailable range: {test.index.min().date()} "
                f"to {test.index.max().date()}"
            )
        else:
            row = test.loc[[date]]
            target_date = date + pd.Timedelta(days=1)

            show_predictions(
                row,
                target_date.date(),
                row["target"].iloc[0]
            )

    except (ValueError, TypeError):
        print("Invalid date. Use YYYY-MM-DD.")


# --------------------------------------------------
# MODE 2: FUTURE-DATE SIMULATION WITH MANUAL INPUTS
# --------------------------------------------------

elif mode == "2":
    date_input = input(
        "Enter a future date (YYYY-MM-DD): "
    ).strip()

    try:
        date = pd.Timestamp(date_input)

        if date <= data.index.max():
            print(
                "Enter a date after the last dataset date: "
                f"{data.index.max().date()}"
            )

        else:
            values = {}

            # Calendar features are derived automatically
            values["month"] = date.month
            values["doy"] = date.dayofyear

            print("\nEnter hypothetical weather values.")
            print("Use the same units as the training dataset.")
            print("All temperature inputs are in degrees Celsius.")

            for feature in features:

                # These features are calculated automatically
                if feature in ("month", "doy"):
                    continue

                if feature == "temp_range":
                    values[feature] = (
                        values["T2M_MAX"] - values["T2M_MIN"]
                    )
                    print(
                        f"Temperature range: "
                        f"{values[feature]:.2f} °C"
                    )
                    continue

                # Request every remaining feature manually
                while True:
                    try:
                        values[feature] = float(
                            input(f"{feature}: ")
                        )
                        break
                    except ValueError:
                        print("Please enter a valid number.")

            # Arrange features in the exact training order
            X = pd.DataFrame([values], columns=features)

            show_predictions(X, date.date())

            print(
                "\nNOTE: These are simulated predictions based on "
                "your hypothetical inputs, not live weather forecasts."
            )

    except (ValueError, TypeError, KeyError):
        print("Invalid date or input.")

else:
    print("Invalid choice. Enter 1 or 2.")
