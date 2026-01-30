# ============================================
# EXTENSIVE KNN PIPELINE FOR POLLUTION_LEVEL
# (ONLY SAVE RESULTS, NO DISPLAY)
# ============================================

import pandas as pd
import numpy as np
import logging
import joblib
import os
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------
# 0. CREATE OUTPUT DIRECTORY
# --------------------------------------------
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------
# 1. LOGGING CONFIGURATION
# --------------------------------------------
logging.basicConfig(
    filename="outputs/knn_pollution_full.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("STARTING EXTENSIVE KNN PIPELINE FOR POLLUTION_LEVEL")

# --------------------------------------------
# 2. LOAD DATA
# --------------------------------------------
df = pd.read_csv("./global-air-pollution-dataset.csv")
logging.info(f"Dataset loaded with shape {df.shape}")

# --------------------------------------------
# 3. CREATE TARGET: Pollution_Level
# --------------------------------------------
def pollution_level(aqi):
    if aqi <= 50:
        return 0
    elif aqi <= 100:
        return 1
    elif aqi <= 150:
        return 2
    else:
        return 3

df["Pollution_Level"] = df["AQI Value"].apply(pollution_level)
logging.info("Pollution_Level target created")

# --------------------------------------------
# 4. FEATURES & TARGET
# --------------------------------------------
X = df[[
    "AQI Value",
    "CO AQI Value",
    "Ozone AQI Value",
    "NO2 AQI Value",
    "PM2.5 AQI Value"
]]
y = df["Pollution_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --------------------------------------------
# 5. DEFINE GRID OF PARAMETERS
# --------------------------------------------
n_neighbors_list = [3,5,7,9,11]
weights_list = ["uniform","distance"]
metrics_list = ["euclidean","manhattan","chebyshev","minkowski"]

param_combinations = list(product(n_neighbors_list, weights_list, metrics_list))
logging.info(f"Total parameter combinations to test: {len(param_combinations)}")

# --------------------------------------------
# 6. TEST ALL COMBINATIONS & STORE RESULTS
# --------------------------------------------
all_results = []

best_acc = 0
best_model = None
best_params = None

for k, w, m in param_combinations:
    knn = KNeighborsClassifier(n_neighbors=k, weights=w, metric=m)
    knn.fit(X_train_s, y_train)
    y_pred = knn.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Save results
    all_results.append({
        "n_neighbors": k,
        "weights": w,
        "metric": m,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    })
    
    logging.info(f"Tested KNN(n_neighbors={k}, weights={w}, metric={m}) -> Accuracy={acc:.4f}")
    
    # Keep track of best model
    if acc > best_acc:
        best_acc = acc
        best_model = knn
        best_params = {"n_neighbors": k, "weights": w, "metric": m}

# --------------------------------------------
# 7. SAVE ALL RESULTS TO CSV
# --------------------------------------------
# Convert dicts to strings for CSV
results_csv = []
for r in all_results:
    results_csv.append({
        "n_neighbors": r["n_neighbors"],
        "weights": r["weights"],
        "metric": r["metric"],
        "accuracy": r["accuracy"],
        "classification_report": str(r["classification_report"]),
        "confusion_matrix": str(r["confusion_matrix"])
    })

results_df = pd.DataFrame(results_csv)
results_df.to_csv("outputs/all_tests.csv", index=False)
logging.info("All parameter combinations saved to outputs/all_tests.csv")

# --------------------------------------------
# 8. SAVE BEST MODEL & SCALER
# --------------------------------------------
joblib.dump(best_model, "outputs/knn_pollution_best_model.pkl")
joblib.dump(scaler, "outputs/scaler_pollution_best.pkl")
logging.info(f"Best model saved with params: {best_params}, Accuracy={best_acc:.4f}")

# --------------------------------------------
# 9. SAVE BEST MODEL METRICS
# --------------------------------------------
best_y_pred = best_model.predict(X_test_s)
best_cm = confusion_matrix(y_test, best_y_pred)
best_report = classification_report(y_test, best_y_pred, output_dict=True)

metrics_summary = {
    "accuracy": best_acc,
    "best_params": best_params,
    "confusion_matrix": best_cm.tolist(),
    "classification_report": best_report
}

import json
with open("outputs/best_model_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

logging.info("Best model metrics saved to outputs/best_model_metrics.json")
logging.info("Pipeline completed successfully. All outputs are in 'outputs/' folder.")
print("✅ Pipeline finished. All outputs saved. No display.")
