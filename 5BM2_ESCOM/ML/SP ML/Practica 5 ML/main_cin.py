import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv("iris.csv")
emails = pd.read_csv("emails.csv")


def ejecutar_practica(df, nombre_dataset, es_texto=False, ax=None):
    print(f"===== DATASET: {nombre_dataset} =====")
    print(f"Dimensiones del dataset: {df.shape}")

    X = df.iloc[:, 1:-1] if es_texto else df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if es_texto:
        X = X.fillna(0)
        X = X.clip(lower=0)
        X = X.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, shuffle=True
    )

    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    modelos = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB()
    }

    tabla_resultados = []

    for nombre, modelo in modelos.items():
        accuracies = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            try:
                modelo.fit(X_tr, y_tr)
                pred = modelo.predict(X_val)
                acc = accuracy_score(y_val, pred)
                accuracies.append(acc)
            except Exception:
                accuracies.append(float('nan'))

        promedio = sum(accuracies) / len(accuracies)
        tabla_resultados.append((nombre, accuracies, promedio))

    print("\n=== Tabla 1 (Validacion Cruzada) ===")
    for modelo, accs, prom in tabla_resultados:
        print(f"Modelo: {modelo}")
        print(f"Accuracies: {accs}")
        print(f"Promedio: {prom}\n")
        print("-" * 40)

    mejor_modelo = max(tabla_resultados, key=lambda x: x[2])[0]
    print(f"Mejor modelo: {mejor_modelo}\n")
    print("-" * 40)

    modelo_final = modelos[mejor_modelo]
    modelo_final.fit(X_train, y_train)
    pred_test = modelo_final.predict(X_test)

    acc_test = accuracy_score(y_test, pred_test)
    print("=== Tabla 2 (Pruebas Finales) ===")
    print(f"Accuracy final: {acc_test}\n")

    print("=== Reporte de Clasificacion ===")
    print(classification_report(y_test, pred_test))
    print("-" * 40)

    cm = confusion_matrix(y_test, pred_test)

    # Crear matriz de confusion
    if ax is not None:
        if nombre_dataset == "iris.csv":
            display_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        else:
            display_labels = ['No Spam', 'Spam']

        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(np.arange(len(display_labels)))
        ax.set_yticks(np.arange(len(display_labels)))
        ax.set_xticklabels(display_labels)
        ax.set_yticklabels(display_labels)
        ax.set_xlabel('Prediccion', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('Real', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_title(f'{nombre_dataset}\nMatriz de Confusion ({mejor_modelo})',
                     fontsize=16, fontweight='bold', pad=20, color='#2C3E50')

        # Agregar valores en las celdas
        for i in range(len(display_labels)):
            for j in range(len(display_labels)):
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return acc_test, mejor_modelo


# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Ejecutar para ambos datasets
acc_iris, modelo_iris = ejecutar_practica(iris, "iris.csv", es_texto=False, ax=ax1)
acc_emails, modelo_emails = ejecutar_practica(emails, "emails.csv", es_texto=True, ax=ax2)

plt.tight_layout()
plt.show()

# Mostrar resumen
print("\n" + "=" * 50)
print("RESUMEN FINAL")
print("=" * 50)
print(f"iris.csv - Mejor modelo: {modelo_iris} - Accuracy: {acc_iris:.4f}")
print(f"emails.csv - Mejor modelo: {modelo_emails} - Accuracy: {acc_emails:.4f}")
print("=" * 50)