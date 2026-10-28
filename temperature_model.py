
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


# --------------------------------------------------
# 1. LOAD AND CLEAN THE NASA POWER DATASET
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / (
    "POWER_Point_Daily_20050101_20241231_012d85N_080d15E_LST.csv"
)

OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}\n"
        "Place the NASA CSV file in the project folder."
    )

# Find the end of NASA's metadata header.
with open(CSV_PATH, encoding="utf-8") as file:
    skip = next(
        (i + 1 for i, line in enumerate(file) if "END HEADER" in line),
        None
    )

if skip is None:
    raise ValueError("Could not find 'END HEADER' in the NASA CSV.")

df = pd.read_csv(CSV_PATH, skiprows=skip)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Build the date column using whichever format is present.
if {"YEAR", "DOY"}.issubset(df.columns):
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(int).astype(str)
        + df["DOY"].astype(int).astype(str).str.zfill(3),
        format="%Y%j"
    )
elif {"YEAR", "MO", "DY"}.issubset(df.columns):
    df["date"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MO"], day=df["DY"])
    )
else:
    raise ValueError("Could not find YEAR/DOY or YEAR/MO/DY columns.")

df = df.sort_values("date").set_index("date")

# NASA uses -999 to indicate missing data.
df = df.replace(-999, np.nan)

print("\nMissing values before interpolation:")
print(df.isna().sum())

# Interpolate using time, then handle any remaining gaps.
df = df.interpolate(method="time").ffill().bfill()

print("\nDuplicate dates:", df.index.duplicated().sum())

# Avoid silently training on duplicated dates.
df = df[~df.index.duplicated(keep="first")]

required = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Required columns missing from CSV: {missing}")


# --------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------

df["month"] = df.index.month
df["doy"] = df.index.dayofyear
df["temp_range"] = df["T2M_MAX"] - df["T2M_MIN"]

df["T2M_lag1"] = df["T2M"].shift(1)
df["T2M_lag2"] = df["T2M"].shift(2)
df["T2M_avg3"] = df["T2M"].rolling(3).mean()
df["T2M_avg7"] = df["T2M"].rolling(7).mean()

df["target"] = df["T2M"].shift(-1)
df = df.dropna()

# Exclude target and original date components from the predictors.
excluded = ["target", "YEAR", "DOY", "MO", "DY"]
features = [col for col in df.columns if col not in excluded]
joblib.dump(features, OUTPUT_DIR / "model_features.pkl")
train = df[df.index < "2023-01-01"]
test = df[df.index >= "2023-01-01"]

if train.empty or test.empty:
    raise ValueError("Training or testing data is empty. Check dataset dates.")

X_train = train[features]
y_train = train["target"]
X_test = test[features]
y_test = test["target"]

print("\nTraining period:", train.index.min().date(), "to", train.index.max().date())
print("Testing period:", test.index.min().date(), "to", test.index.max().date())
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Features:", features)

# Save cleaned data.
df.to_csv(OUTPUT_DIR / "cleaned_data.csv")
print("\nCleaned data saved to results/cleaned_data.csv")


# --------------------------------------------------
# 3. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

baseline_predictions = test["T2M"]

print("\n--- Baseline: tomorrow = today ---")
print("R2:", round(r2_score(y_test, baseline_predictions), 3))
print("MAE:", round(mean_absolute_error(y_test, baseline_predictions), 3), "°C")

models = [
    ("Linear Regression", LinearRegression()),
    (
        "Random Forest",
        RandomForestRegressor(n_estimators=200, random_state=42)
    ),
    (
        "XGBoost",
        XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
    ),
]

trained_models = {}
predictions = {}

print("\n--- Model results: next-day prediction ---")


for name, model in models:
    model.fit(X_train, y_train)

    # Save the trained model
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, OUTPUT_DIR / filename)

    pred = model.predict(X_test)

    trained_models[name] = model
    predictions[name] = pred

    print(
        f"{name}: "
        f"R2 = {r2_score(y_test, pred):.3f}, "
        f"MAE = {mean_absolute_error(y_test, pred):.3f} °C"
    )



# --------------------------------------------------
# 4. ACTUAL VS PREDICTED PLOT
# --------------------------------------------------

xgb_predictions = predictions["XGBoost"]

plt.figure(figsize=(10, 4))
plt.plot(test.index[-120:], y_test.iloc[-120:], label="Actual")
plt.plot(test.index[-120:], xgb_predictions[-120:], label="XGBoost predicted")
plt.title("Actual vs predicted temperature")
plt.ylabel("Temperature (°C)")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=200)
plt.close()


# --------------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.Series(
    trained_models["XGBoost"].feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(8, 6))
importance.plot(kind="barh", title="XGBoost feature importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
plt.close()

print("\nTop 10 features:")
print(importance.sort_values(ascending=False).head(10))


# --------------------------------------------------
# 6. SEASONAL TEMPERATURE PATTERN
# --------------------------------------------------

df.groupby("month")["T2M"].mean().plot(
    kind="bar",
    figsize=(8, 4),
    title="Average temperature by month"
)
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "seasonal_pattern.png", dpi=200)
plt.close()


# --------------------------------------------------
# 7. COMPARE 1-, 3-, AND 7-DAY PREDICTIONS
# --------------------------------------------------

horizons = [1, 3, 7]
baseline_mae = []
xgb_mae = []

print("\n--- Multi-day prediction results ---")

for horizon in horizons:
    horizon_df = df.copy()
    horizon_df["target"] = horizon_df["T2M"].shift(-horizon)
    horizon_df = horizon_df.dropna()

    tr = horizon_df[horizon_df.index < "2023-01-01"]
    te = horizon_df[horizon_df.index >= "2023-01-01"]

    if tr.empty or te.empty:
        raise ValueError(f"Insufficient data for the {horizon}-day prediction.")

    f = [col for col in horizon_df.columns if col not in excluded]

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )

    model.fit(tr[f], tr["target"])
    pred = model.predict(te[f])

    base_error = mean_absolute_error(te["target"], te["T2M"])
    model_error = mean_absolute_error(te["target"], pred)

    baseline_mae.append(base_error)
    xgb_mae.append(model_error)

    print(
        f"{horizon}-day ahead | "
        f"Baseline MAE: {base_error:.3f} °C | "
        f"XGBoost MAE: {model_error:.3f} °C"
    )

x = np.arange(len(horizons))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5))
b1 = ax.bar(x - width / 2, baseline_mae, width, label="Baseline")
b2 = ax.bar(x + width / 2, xgb_mae, width, label="XGBoost")

ax.bar_label(b1, fmt="%.2f", padding=2)
ax.bar_label(b2, fmt="%.2f", padding=2)
ax.set_xticks(x, [f"{h}-day ahead" for h in horizons])
ax.set_ylabel("Mean Absolute Error (°C)")
ax.set_title("Prediction error: baseline vs XGBoost")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "horizon_comparison.png", dpi=200)
plt.close(fig)


# --------------------------------------------------
# 8. COMPACT LINEAR REGRESSION EQUATION
# --------------------------------------------------

small_features = ["T2M", "T2M_avg3", "T2M_avg7", "month", "RH2M"]

compact_model = LinearRegression()
compact_model.fit(train[small_features], train["target"])
joblib.dump(compact_model, OUTPUT_DIR / "compact_model.pkl")
joblib.dump(small_features, OUTPUT_DIR / "compact_features.pkl")

compact_predictions = compact_model.predict(test[small_features])

print("\n--- Compact Linear Regression model ---")
print(
    "Test R2:",
    round(r2_score(test["target"], compact_predictions), 3)
)

print("\nTomorrow's temperature equation:")
for name, coefficient in zip(small_features, compact_model.coef_):
    print(f"  {coefficient:+.4f} × {name}")

print(f"  Intercept: {compact_model.intercept_:+.4f}")


def predict_tomorrow(today, avg_3day, avg_7day, month, humidity):
    inputs = pd.DataFrame(
        [[today, avg_3day, avg_7day, month, humidity]],
        columns=small_features
    )
    return float(compact_model.predict(inputs)[0])


example = predict_tomorrow(
    today=29.5,
    avg_3day=29.2,
    avg_7day=28.8,
    month=5,
    humidity=70
)

print(f"\nExample prediction: {example:.2f} °C")


# --------------------------------------------------
# 9. FINAL SUMMARY
# --------------------------------------------------

print("\nAll tasks completed.")
print("Plots and cleaned data saved in:", OUTPUT_DIR)
print("Generated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)
