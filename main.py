"""
CC3074 - Minería de Datos
Laboratorio 6: KNN

"""

# ================================
# IMPORTS
# ================================
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report
)

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

try:
    import pyreadr
except ModuleNotFoundError:
    pyreadr = None


# ================================
# CARGA DE DATOS
# ================================
RDATA_PATH = "./data/listings.RData"

df_raw = None
if pyreadr is not None:
    result = pyreadr.read_r(RDATA_PATH)
    df_raw = next(iter(result.values()))

df = df_raw.copy()


# ================================
# PREPROCESAMIENTO
# ================================
df["price"] = (
    df["price"].astype(str)
    .str.replace(r"[$,]", "", regex=True)
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df = df[(df["price"] > 0) & (df["price"] <= 1000)]
df.dropna(subset=["price"], inplace=True)

# SOLO VARIABLES NUMERICAS
df_num = df.select_dtypes(include=[np.number]).copy()

y = df_num["price"]
X = df_num.drop(columns=["price"])

X = X.fillna(X.median())


# ================================
# VARIABLE CATEGORICA
# ================================
q33 = df["price"].quantile(0.33)
q66 = df["price"].quantile(0.66)

def clasificar(p):
    if p <= q33:
        return "Economica"
    elif p <= q66:
        return "Intermedia"
    else:
        return "Cara"

df["precio_cat"] = df["price"].apply(clasificar)


# ================================
# SPLIT (MISMO QUE LAB 5)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_cls = X.copy()
y_cls = df["precio_cat"]

X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)


# ============================================================
# ACTIVIDAD 1 — KNN REGRESION
# ============================================================
print("\nACTIVIDAD 1 — KNN REGRESIÓN")

pipeline_knn_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

pipeline_knn_reg.fit(X_train, y_train)
y_pred_knn = pipeline_knn_reg.predict(X_test)


# ============================================================
# ACTIVIDAD 2 — EVALUACION
# ============================================================
print("\nACTIVIDAD 2 — EVALUACIÓN")

rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
mae = mean_absolute_error(y_test, y_pred_knn)
r2 = r2_score(y_test, y_pred_knn)

print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)


# ============================================================
# ACTIVIDAD 4 — KNN CLASIFICACION
# ============================================================
print("\nACTIVIDAD 4 — KNN CLASIFICACIÓN")

pipeline_knn_cls = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

pipeline_knn_cls.fit(X_cls_train, y_cls_train)
y_pred_cls = pipeline_knn_cls.predict(X_cls_test)


# ============================================================
# ACTIVIDAD 5 — EVALUACION CLASIFICACION
# ============================================================
print("\nACTIVIDAD 5 — EVALUACIÓN CLASIFICACIÓN")

acc = accuracy_score(y_cls_test, y_pred_cls)

print("Accuracy:", acc)
print("\nReporte:")
print(classification_report(y_cls_test, y_pred_cls))