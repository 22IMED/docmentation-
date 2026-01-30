import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Chargement du dataset
# ---------------------------------------------------------
df = pd.read_csv("global air pollution dataset.csv")

print("Colonnes disponibles :")
print(df.columns.tolist())

# ---------------------------------------------------------
# 2. Définition de la variable cible
# ---------------------------------------------------------
target = "AQI Category"

if target not in df.columns:
    raise ValueError(f"La colonne '{target}' n'existe pas dans le dataset.")

# On supprime les lignes sans catégorie AQI
df = df.dropna(subset=[target])

# ---------------------------------------------------------
# 3. Séparation X / y
# ---------------------------------------------------------
X = df.drop(columns=[target])
y = df[target]

# Détection automatique des colonnes
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numériques :", numeric_cols)
print("Catégorielles :", categorical_cols)

# ---------------------------------------------------------
# 4. Préprocessing
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ---------------------------------------------------------
# 5. Modèle Random Forest
# ---------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", clf)
])

# ---------------------------------------------------------
# 6. Train / Test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 7. Entraînement
# ---------------------------------------------------------
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------
# 8. Évaluation
# ---------------------------------------------------------
y_pred = pipeline.predict(X_test)

print("\n=== Évaluation du modèle Random Forest ===")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nClassification report :")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 9. Exemple de prédiction
# ---------------------------------------------------------
example = X.iloc[[0]]  # première ligne du dataset
pred_example = pipeline.predict(example)[0]

print("\nExemple de prédiction sur la première ligne :")
print("Valeur réelle :", y.iloc[0])
print("Prédiction    :", pred_example)
