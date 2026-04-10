"""
CC3074 - Minería de Datos
Laboratorio 6: KNN
Integrantes:
  - Mia Alejandra Fuentes Merida, 23775
  - María José Girón Isidro, 23559
  - Leonardo Dufrey Mejía Mejía, 23648
"""

# IMPORTS
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

import pyreadr


# CARGA DE DATOS
RDATA_PATH = "./data/listings.RData"

result = pyreadr.read_r(RDATA_PATH)
df = next(iter(result.values()))


# PREPROCESAMIENTO
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


# VARIABLE CATEGORICA
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

# SPLIT (MISMO QUE LAB 5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_cls = X.copy()
y_cls = df["precio_cat"]

X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

# ACTIVIDAD 1 — KNN REGRESIÓN
print("\n" + "="*60)
print("ACTIVIDAD 1 — KNN REGRESIÓN")
print("="*60)

pipeline_knn_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

t0 = time.time()
pipeline_knn_reg.fit(X_train, y_train)
tiempo_knn = time.time() - t0

y_pred_knn = pipeline_knn_reg.predict(X_test)

# ACTIVIDAD 2 — EVALUACIÓN REGRESIÓN
print("\n" + "="*60)
print("ACTIVIDAD 2 — EVALUACIÓN REGRESIÓN")
print("="*60)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_knn))
mae = mean_absolute_error(y_test, y_pred_knn)
r2 = r2_score(y_test, y_pred_knn)

print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")
print(f"Tiempo: {tiempo_knn:.4f}s")

# ACTIVIDAD 3 — COMPARACIÓN
print("\n" + "="*60)
print("ACTIVIDAD 3 — COMPARACIÓN")
print("="*60)

comp = pd.DataFrame({
    "Modelo": [
        "Regresión Lineal (Ridge)",
        "Árbol Regresión",
        "Random Forest",
        "Naive Bayes",
        "KNN"
    ],
    "RMSE": [
        138.04,
        122.66,
        105.10,
        556.50,
        rmse
    ],
    "R2": [
        0.4127,
        0.5363,
        0.6595,
        -8.545,
        r2
    ]
})

print(comp)

# ACTIVIDAD 4 — KNN CLASIFICACIÓN

print("\n" + "="*60)
print("ACTIVIDAD 4 — KNN CLASIFICACIÓN")
print("="*60)

pipeline_knn_cls = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

t0 = time.time()
pipeline_knn_cls.fit(X_cls_train, y_cls_train)
tiempo_knn_cls = time.time() - t0

y_pred_cls = pipeline_knn_cls.predict(X_cls_test)

# ACTIVIDAD 5 — EVALUACIÓN CLASIFICACIÓN
print("\n" + "="*60)
print("ACTIVIDAD 5 — EVALUACIÓN CLASIFICACIÓN")
print("="*60)

acc = accuracy_score(y_cls_test, y_pred_cls)

print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Tiempo: {tiempo_knn_cls:.4f}s")

print("\nReporte de clasificación:")
print(classification_report(y_cls_test, y_pred_cls))

# GRÁFICA — REGRESIÓN
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_knn, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("KNN Regresión")
plt.tight_layout()
plt.show()

# ACTIVIDAD 5 — EVALUACIÓN CLASIFICACIÓN
print("\n" + "="*60)
print("ACTIVIDAD 5 — EVALUACIÓN CLASIFICACIÓN")
print("="*60)

acc = accuracy_score(y_cls_test, y_pred_cls)

print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Tiempo: {tiempo_knn_cls:.4f}s")

print("\nReporte de clasificación:")
print(classification_report(y_cls_test, y_pred_cls))

# GRÁFICA — REGRESIÓN
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_knn, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("KNN Regresión")
plt.tight_layout()
plt.show()

# ACTIVIDAD 6 — MATRIZ DE CONFUSIÓN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("\n" + "="*60)
print("ACTIVIDAD 6 — MATRIZ DE CONFUSIÓN")
print("="*60)

labels = ["Economica", "Intermedia", "Cara"]
cm = confusion_matrix(y_cls_test, y_pred_cls, labels=labels)

print("\nMatriz de confusión:")
print(pd.DataFrame(cm, index=labels, columns=labels))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión — KNN")
plt.show()

# ACTIVIDAD 7 — SOBREAJUSTE
print("\n" + "="*60)
print("ACTIVIDAD 7 — SOBREAJUSTE")
print("="*60)

acc_train = accuracy_score(y_cls_train, pipeline_knn_cls.predict(X_cls_train))

print(f"Accuracy entrenamiento: {acc_train:.4f}")
print(f"Accuracy prueba:        {acc:.4f}")
print(f"Diferencia:             {acc_train - acc:+.4f}")

# ACTIVIDAD 8 — VALIDACIÓN CRUZADA
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("\n" + "="*60)
print("ACTIVIDAD 8 — VALIDACIÓN CRUZADA")
print("="*60)

cv_scores = cross_val_score(
    pipeline_knn_cls,
    X_cls_train,
    y_cls_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy"
)

print("Accuracy por fold:", np.round(cv_scores, 4))
print(f"Accuracy promedio: {cv_scores.mean():.4f}")
print(f"Desviación estándar: {cv_scores.std():.4f}")


# ACTIVIDAD 9 — TUNEO DE HIPERPARÁMETROS
from sklearn.model_selection import GridSearchCV

print("\n" + "="*60)
print("ACTIVIDAD 9 — TUNEO KNN")
print("="*60)

param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2] 
}

grid = GridSearchCV(
    pipeline_knn_cls,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_cls_train, y_cls_train)

print("Mejores parámetros:", grid.best_params_)
print(f"Mejor accuracy CV: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
y_pred_tuned = best_model.predict(X_cls_test)

acc_tuned = accuracy_score(y_cls_test, y_pred_tuned)

print(f"Accuracy base:  {acc:.4f}")
print(f"Accuracy tuned: {acc_tuned:.4f}")
print(f"Mejora: {acc_tuned - acc:+.4f}")
