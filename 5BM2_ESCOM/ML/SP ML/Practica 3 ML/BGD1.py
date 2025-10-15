import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def obtencion_de_w(x, y, paso=0.01, w=None):
    if w is None:
        w = [0] * len(x.columns)

    n = len(x)
    nw = len(w)

    for i in range(nw):
        calc = 0
        for j in range(n):
            y_pred_j = w[i] * x.iloc[j, i]
            calc += (y_pred_j - y.iloc[j, 0]) * x.iloc[j, i]

        w[i] -= float(2 * paso * calc)
    return w


def bgd_artesanal(x_train, y_train, x_test, y_test, paso=0.01, w=None):
    wn = obtencion_de_w(x_train, y_train, paso=paso, w=w)

    nw = len(wn)
    n = len(x_test)

    y_pred = [0] * n
    error_estimacion = 0
    for i in range(n):
        y_pred[i] = float(sum(x_test.iloc[i, j] * wn[j] for j in range(nw)))
        error_estimacion += abs(y_pred[i] - float(y_test.iloc[i, 0]))

    return y_pred, error_estimacion, wn


def bgd_artesanal_grafico(x_train, y_train, x_test, y_test, paso=0.01, w=None, iteraciones=1):
    import matplotlib.pyplot as plt

    y_pred_list = []
    error_list = []
    w_list = []

    # Inicialización de pesos
    if w is None:
        w = [0] * len(x_train.columns)

    for i in range(iteraciones):
        y_pred, error_estimacion, w = bgd_artesanal(x_train, y_train, x_test, y_test, paso=paso, w=w)
        y_pred_list.append(y_pred)
        error_list.append(error_estimacion)
        w_list.append(w.copy())

    # === Impresiones de resultados ===
    print("Valores de w por iteración:")
    for i, wi in enumerate(w_list):
        print(f"Iteración {i}: {wi}")

    print("\nValores de y reales:\n", y_test.T)
    print("\nValores de y_pred:")
    for i, ypi in enumerate(y_pred_list):
        print(f"Iteración {i}: {ypi}")

    print("\nErrores de estimación:")
    for i, err in enumerate(error_list):
        print(f"Iteración {i}: {err}")

    # === GRAFICAS ===
    plt.style.use('default')  # estilo limpio base
    plt.rcParams.update({
        'font.size': 11,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'grid.color': 'gray',
        'grid.alpha': 0.3
    })

    # --- 1️⃣ Relación Precio (MDP) vs Terreno (m²) ---
    plt.figure(figsize=(9, 6))

    # Ordenamos para líneas limpias
    x_sorted, y_sorted = zip(*sorted(zip(x_test.iloc[:, 0], y_test.iloc[:, 0])))

    # Datos reales
    plt.scatter(x_sorted, y_sorted, color='black', label='Datos reales', s=40, alpha=0.8)
    plt.plot(x_sorted, y_sorted, color='black', linewidth=1.8, alpha=0.7)

    # Colores suaves para iteraciones
    colores = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    for i in range(min(iteraciones, len(colores))):
        x_pred_sorted, y_pred_sorted = zip(*sorted(zip(x_test.iloc[:, 0], y_pred_list[i])))
        plt.plot(x_pred_sorted, y_pred_sorted, color=colores[i], linewidth=2,
                 label=f'Iteración {i+1}', alpha=0.9)
        plt.scatter(x_pred_sorted, y_pred_sorted, color=colores[i], s=35, alpha=0.8, edgecolors='white')

    plt.title("Evolución del modelo: Precio (MDP) vs Terreno (m²)", fontsize=14, fontweight='bold')
    plt.xlabel("Terreno (m²)", fontsize=12)
    plt.ylabel("Precio (MDP)", fontsize=12)
    plt.legend(frameon=False, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --- 2️⃣ Error entre iteraciones ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, iteraciones + 1), error_list, marker='o', color='#333333',
             linewidth=2, markersize=7)
    plt.title("Error de estimación por iteración", fontsize=14, fontweight='bold')
    plt.xlabel("Iteración", fontsize=12)
    plt.ylabel("Error total (MDP)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return y_pred_list, error_list, w_list



data_x = pd.read_csv("casas.csv", usecols=[0])
data_y = pd.read_csv("casas.csv", usecols=[1])

# Convertir y a DataFrame si es Series
if isinstance(data_y, pd.Series):
    data_y = data_y.to_frame()

print("Features (Terreno):\n", data_x.T, "\n\ntargets (Precio):\n", data_y.T)

x_train, x_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.3, train_size=0.7, shuffle=True, random_state=0
)
x_train, x_test, y_train, y_test = (
    x_train.astype(float),
    x_test.astype(float),
    y_train.astype(float),
    y_test.astype(float),
)

print("\n\nTraining set\nFeatures:\n", x_train.T, "\n\nTargets:\n", y_train.T)
print("\nTest set\nFeatures:\n", x_test.T, "\n\nTargets:\n", y_test.T, "\n\nProceso para 5 iteraciones")

bgd_artesanal_grafico(x_train, y_train, x_test, y_test, paso=7e-8, iteraciones=4)
