import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------------------------------

def validar_modelo(X_train, y_train, modelo, k=3):
    """Realiza validación cruzada con KFold y devuelve accuracies."""
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    accuracies = []
    for train_index, val_index in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        modelo.fit(X_tr, y_tr)
        y_pred = modelo.predict(X_val)
        accuracies.append(accuracy_score(y_val, y_pred))
    return accuracies, np.mean(accuracies)


def probar_modelo_final(modelo, X_train, y_train, X_test, y_test, dataset_name, ax=None):
    """Entrena el modelo y muestra matriz de confusión y reporte de clasificación."""
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"RESULTADOS FINALES: {dataset_name}")
    print(f"Accuracy de prueba: {acc:.4f}")
    print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Graficar en subplot si se proporciona un eje
    if ax is not None:
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Matriz de confusión - {dataset_name}")
        ax.grid(False)
    return acc


# ----------------------------------------------------------
# PROCESO PARA IRIS
# ----------------------------------------------------------
print("\n=== Procesando dataset: iris.csv ===")

iris = pd.read_csv("iris.csv")

X_iris = iris.iloc[:, :-1]
y_iris = iris.iloc[:, -1]

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=0, shuffle=True
)

resultados_iris = {"Distribución": [], "Pliegue": [], "Accuracy": []}

for modelo, nombre in [(GaussianNB(), "Normal"), (MultinomialNB(), "Multinomial")]:
    accuracies, promedio = validar_modelo(X_train_i, y_train_i, modelo)
    for i, acc in enumerate(accuracies, start=1):
        resultados_iris["Distribución"].append(nombre)
        resultados_iris["Pliegue"].append(i)
        resultados_iris["Accuracy"].append(acc)
    resultados_iris["Distribución"].append(nombre)
    resultados_iris["Pliegue"].append("Promedio")
    resultados_iris["Accuracy"].append(promedio)

tabla_iris = pd.DataFrame(resultados_iris)
print("\nTabla 1 – Validación cruzada (iris.csv):")
print(tabla_iris)

mejor_modelo_iris = GaussianNB() if tabla_iris.iloc[3, 2] > tabla_iris.iloc[7, 2] else MultinomialNB()

# ----------------------------------------------------------
# PROCESO PARA EMAILS (sin eliminar filas)
# ----------------------------------------------------------
print("\n=== Procesando dataset: emails.csv ===")

emails = pd.read_csv("emails.csv")

# Limpieza sin eliminar filas
y_raw = emails.iloc[:, -1].astype(str).str.strip().str.lower()

y_emails = y_raw.replace({
    'spam': 1, 'ham': 0, 'not spam': 0, 'non-spam': 0,
    '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'true': 1, 'false': 0,
    'nan': 0, '': 0
})

# Intentar conversión numérica, NaN => 0
y_emails = pd.to_numeric(y_emails, errors='coerce').fillna(0).astype(int)

# Mantener todas las filas
X_emails = emails.iloc[:, 1:-1].fillna(0)

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_emails, y_emails, test_size=0.3, random_state=0, shuffle=True
)

resultados_emails = {"Distribución": [], "Pliegue": [], "Accuracy": []}

for modelo, nombre in [(GaussianNB(), "Normal"), (MultinomialNB(), "Multinomial")]:
    accuracies, promedio = validar_modelo(X_train_e, y_train_e, modelo)
    for i, acc in enumerate(accuracies, start=1):
        resultados_emails["Distribución"].append(nombre)
        resultados_emails["Pliegue"].append(i)
        resultados_emails["Accuracy"].append(acc)
    resultados_emails["Distribución"].append(nombre)
    resultados_emails["Pliegue"].append("Promedio")
    resultados_emails["Accuracy"].append(promedio)

tabla_emails = pd.DataFrame(resultados_emails)
print("\nTabla 1 – Validación cruzada (emails.csv):")
print(tabla_emails)

mejor_modelo_emails = GaussianNB() if tabla_emails.iloc[3, 2] > tabla_emails.iloc[7, 2] else MultinomialNB()

# ----------------------------------------------------------
# VISUALIZAR AMBAS MATRICES EN UNA SOLA FIGURA
# ----------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
acc_iris = probar_modelo_final(mejor_modelo_iris, X_train_i, y_train_i, X_test_i, y_test_i, "iris.csv", ax1)
acc_emails = probar_modelo_final(mejor_modelo_emails, X_train_e, y_train_e, X_test_e, y_test_e, "emails.csv", ax2)

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# TABLA 2: Resultados de pruebas finales
# ----------------------------------------------------------
tabla_final = pd.DataFrame({
    "Dataset": ["iris.csv", "emails.csv"],
    "Distribución": ["GaussianNB" if isinstance(mejor_modelo_iris, GaussianNB) else "MultinomialNB",
                     "GaussianNB" if isinstance(mejor_modelo_emails, GaussianNB) else "MultinomialNB"],
    "Accuracy": [acc_iris, acc_emails]
})

print("\nTabla 2 – Resultados finales (Pruebas):")
print(tabla_final)
